import os
import torch
import logging
import numpy as np
import torch.multiprocessing as mp
from typing import Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from certreach.learning.training import train
from certreach.learning.networks import SingleBVPNet, NetworkConfig
from certreach.common.matlab_loader import load_matlab_data, compare_with_nn
from certreach.common.dataset import ReachabilityDataset

from .loss import initialize_loss, initialize_hamiltonian
from examples.utils.experiment_utils import get_experiment_folder
from examples.factories import register_example

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)

logger = logging.getLogger(__name__)  # Move logger to module level

def double_integrator_boundary(states, radius=0.25):
    """Compute boundary values for both PyTorch tensors and dReal variables."""
    # Check if using PyTorch tensors
    using_torch = isinstance(states, torch.Tensor)
    
    if using_torch:
        return torch.norm(states, dim=1, keepdim=True) - radius
    else:
        # dReal mode - coords will be a list of variables [t, x, v]
        x, v = states[0], states[1]
        # Manual computation of L2 norm for dReal
        return (x * x + v * v)**0.5 - radius

@register_example
class DoubleIntegrator:
    Name = "double_integrator"
    DEFAULT_MATLAB_FILE = "value_function.mat"
    NUM_STATES = 2  # Add this class variable to define number of states

    def __init__(self, args):
        self.args = args
        self.root_path = get_experiment_folder(args.logging_root, self.Name)
        self.device = torch.device(args.device)
        
        # Initialize input bounds
        self.input_bounds = {
            'min': torch.tensor([-args.input_max]),
            'max': torch.tensor([args.input_max])
        }
                
        # Initialize model and other components only when needed
        self.model = None
        self.loss_fn = None
        self.hamiltonian_fn = None
        self.boundary_fn = double_integrator_boundary

    def initialize_components(self):
        """Initialize dataset, model, and loss function"""
               
        # Initialize model if needed
        if self.model is None:
            config = NetworkConfig(
                in_features=self.args.in_features,
                out_features=self.args.out_features,
                hidden_features=self.args.num_nl,
                num_hidden_layers=self.args.num_hl,
                activation_type='sine',  # Default to sine activation
                use_polynomial=self.args.use_polynomial,
                poly_degree=self.args.poly_degree
            )
            
            self.model = SingleBVPNet(config=config).to(self.device)

        if self.loss_fn is None:
            self.loss_fn = initialize_loss(
                self.device,
                input_bounds=self.input_bounds,
                minWith=self.args.minWith,
                reachMode=self.args.reachMode,
                reachAim=self.args.reachAim
            )

        if self.hamiltonian_fn is None:
            self.hamiltonian_fn = initialize_hamiltonian(
                self.device,
                input_bounds=self.input_bounds,
                minWith=self.args.minWith,
                reachMode=self.args.reachMode,
                reachAim=self.args.reachAim
            )

    def train(self):
        """Train the model"""
        self.initialize_components()

        dataset = ReachabilityDataset(
            batch_size=self.args.batch_size,
            tMin=self.args.tMin,
            tMax=self.args.tMax,
            seed=self.args.seed,
            device=self.device,
            num_states=self.NUM_STATES,
            compute_boundary_values=self.boundary_fn,  # Use class boundary_fn
            percentage_in_counterexample=self.args.percentage_in_counterexample,
            percentage_at_t0=self.args.percentage_at_t0,
            epsilon_radius=self.args.epsilon_radius
        )
        
        train(
            model=self.model,
            dataset=dataset,
            epochs=self.args.num_epochs,
            lr=self.args.lr,
            epochs_til_checkpoint=self.args.epochs_til_ckpt,
            model_dir=self.root_path,
            loss_fn=self.loss_fn,
            pretrain_percentage=self.args.pretrain_percentage,
            time_min=self.args.tMin,
            time_max=self.args.tMax,
            validation_fn=self.validate,
            device=self.device,
            use_amp=True  # New parameter with default
        )

    def validate(self, model, ckpt_dir, epoch, tmax=None):
        """Validation function called during training"""

        # Define evaluation time points
        if tmax is None or tmax==self.args.tMax:
            times = [self.args.tMin, 0.5*(self.args.tMin+self.args.tMax), self.args.tMax]
        else:
            times = [self.args.tMin, tmax, self.args.tMax]
        num_times = len(times)

        # Create state space sampling grid
        state_range = torch.linspace(-1, 1, 200)
        fig, axes = plt.subplots(num_times, 1, figsize=(10, 15))

        for i, t in enumerate(times):
            X, V = torch.meshgrid(state_range, state_range, indexing='ij')
            positions = X.reshape(-1, 1)
            velocities = V.reshape(-1, 1)
            time_coords = torch.ones_like(positions) * t

            coords = torch.cat((time_coords, positions, velocities), dim=1).to(self.device)
            model_in = {'coords': coords}
            model_out = model(model_in)['model_out'].detach().cpu().numpy()
            model_out = model_out.reshape(X.shape)

            # Create filled contour plot
            contour = axes[i].contourf(X, V, model_out, levels=50, cmap='bwr')
            
            # Add zero level set
            zero_contour = axes[i].contour(X, V, model_out, levels=[0], colors='k', linewidths=2)
            axes[i].clabel(zero_contour, inline=True, fontsize=8)
            
            axes[i].set_title(f"t = {t:.2f}")
            axes[i].set_xlabel("Position (x)")
            axes[i].set_ylabel("Velocity (v)")
            fig.colorbar(contour, ax=axes[i])

        plt.tight_layout()
        fig.savefig(os.path.join(ckpt_dir, f'DoubleIntegrator_val_epoch_{epoch:04d}.png'))
        plt.close(fig)

    def plot_final_model(self, model, save_dir, epsilon, save_file="Final_Model_Comparison_With_Zero_Set.png"):
        """Plot comparison of value functions"""
        state_range = torch.linspace(-1, 1, 200)
        X, V = torch.meshgrid(state_range, state_range, indexing='ij')
        positions = X.reshape(-1, 1)
        velocities = V.reshape(-1, 1)
        time_coords = torch.ones_like(positions) * self.args.tMax

        # Move coordinates to the same device as the model
        coords = torch.cat((time_coords, positions, velocities), dim=1).to(self.device)
        model_in = {'coords': coords}
        
        # Get output and move back to CPU for plotting
        model_out = model(model_in)['model_out'].cpu().detach().numpy().reshape(X.shape)
        adjusted_model_out = model_out - epsilon

        # Create comparison plots
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))

        # Original Value Function
        contour1 = axes[0].contourf(X, V, model_out, levels=50, cmap='bwr')
        zero_level1 = axes[0].contour(X, V, model_out, levels=[0], colors='k', linewidths=2)
        axes[0].clabel(zero_level1, inline=True, fontsize=8)
        axes[0].set_title("Original Value Function")
        axes[0].set_xlabel("Position (x)")
        axes[0].set_ylabel("Velocity (v)")
        fig.colorbar(contour1, ax=axes[0])

        # Epsilon-Adjusted Value Function
        contour2 = axes[1].contourf(X, V, adjusted_model_out, levels=50, cmap='bwr')
        zero_level2 = axes[1].contour(X, V, adjusted_model_out, levels=[0], colors='k', linewidths=2)
        axes[1].clabel(zero_level2, inline=True, fontsize=8)
        axes[1].set_title(f"Epsilon-Adjusted Value ($\epsilon$={epsilon})")
        axes[1].set_xlabel("Position (x)")
        axes[1].set_ylabel("Velocity (v)")
        fig.colorbar(contour2, ax=axes[1])

        # Zero-Level Set Comparison
        axes[2].contour(X, V, model_out, levels=[0], colors='b', linewidths=2, linestyles='--')
        axes[2].contour(X, V, adjusted_model_out, levels=[0], colors='r', linewidths=2, linestyles='-')
        axes[2].set_title("Zero-Level Set Comparison")
        axes[2].set_xlabel("Position (x)")
        axes[2].set_ylabel("Velocity (v)")
        axes[2].legend(["Original", f"Epsilon-Adjusted ($\epsilon$={epsilon})"], loc="upper right")

        plt.tight_layout()
        save_path = os.path.join(save_dir, save_file)
        plt.savefig(save_path)
        plt.close(fig)
        logger.debug(f"Saved comparison plot at: {save_path}")

    def compare_with_true_values(
        self, 
        matlab_file_path: Optional[str] = None, 
        visualize: bool = True
    ):
        """
        Compare neural network predictions with true value function from MATLAB.
        
        Args:
            matlab_file_path: Optional path to the .mat file. If None, uses default file in same directory
            visualize: Whether to plot the comparison
            
        Returns:
            Tuple of (difference array, mean squared error)
        """
        if matlab_file_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            matlab_file_path = os.path.join(current_dir, self.DEFAULT_MATLAB_FILE)

        logger.debug("Loading true value function from: %s", matlab_file_path)
        
        if not os.path.exists(matlab_file_path):
            logger.error("MATLAB file not found: %s", matlab_file_path)
            raise FileNotFoundError(
                f"MATLAB file not found at: {matlab_file_path}. "
                f"Please ensure '{self.DEFAULT_MATLAB_FILE}' is in the same directory as double_integrator.py"
            )
            
        if self.model is None:
            raise ValueError("Neural network model not initialized")
            
        # Load MATLAB data
        matlab_data = load_matlab_data(matlab_file_path)
        
        # Compare with neural network
        save_path = os.path.join(self.root_path, 'DoubleIntegrator_true_value_comparison.png')
        difference, mse = compare_with_nn(
            self.model,
            matlab_data,
            visualize=visualize,
            save_path=save_path
        )
        
        # Log results
        logger.info("Comparison Results:")
        logger.info("Mean Squared Error: %.6f", mse)
        logger.info("Max Absolute Error: %.6f", np.max(np.abs(difference)))
            
        return difference, mse
