import os
import torch
import logging
import numpy as np
import torch.multiprocessing as mp
from typing import Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from certreach.common.dataset import ReachabilityDataset
from certreach.learning.training import train
from certreach.learning.networks import SingleBVPNet
from certreach.verification.symbolic import extract_symbolic_model
from certreach.verification.dreal_utils import (
    extract_dreal_partials,
    process_dreal_result
)
from certreach.verification.verify import verify_system
from certreach.common.matlab_loader import load_matlab_data, compare_with_nn
from .verification import dreal_double_integrator_BRS
from .loss import initialize_loss
from examples.utils.experiment_utils import get_experiment_folder, save_experiment_details
from examples.factories import register_example

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)

def double_integrator_boundary(coords, radius=0.25):
        pos = coords[:, 1:3]  # Extract [x, v]
        boundary_values = torch.norm(pos, dim=1, keepdim=True)
        return boundary_values - radius

@register_example
class DoubleIntegrator:
    Name = "double_integrator"
    DEFAULT_MATLAB_FILE = "value_function.mat"  # Add this line

    def __init__(self, args):
        self.args = args
        self.root_path = get_experiment_folder(args.logging_root, self.Name)
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(args.device)
                
        # Initialize model and other components only when needed
        self.model = None
        self.dataset = None
        self.loss_fn = None
        self.verification_fn = dreal_double_integrator_BRS  # Add this line

    def initialize_components(self, counterexample: Optional[torch.Tensor] = None):
        """Initialize dataset, model, and loss function"""
        # Initialize dataset if it doesn't exist
        if self.dataset is None:
            self.dataset = ReachabilityDataset(
                numpoints=85000,
                tMin=self.args.tMin,
                tMax=self.args.tMax,
                pretrain=self.args.pretrain,
                pretrain_iters=self.args.pretrain_iters,
                counter_start=self.args.counter_start,
                counter_end=self.args.counter_end,
                num_src_samples=self.args.num_src_samples,
                seed=self.args.seed,
                device=self.device,
                num_states=2,  # [position, velocity]
                compute_boundary_values=double_integrator_boundary
            )
        
        # Add counterexample if provided
        if counterexample is not None:
            self.dataset.add_counterexample(counterexample)
               
        # Initialize model if needed
        if self.model is None:
            self.model = SingleBVPNet(
                in_features=self.args.in_features,
                out_features=self.args.out_features,
                type=self.args.model_type,  # Changed from model to model_type
                mode=self.args.model_mode,  # Changed from mode to model_mode
                hidden_features=self.args.num_nl,
                num_hidden_layers=self.args.num_hl,
                use_polynomial=self.args.use_polynomial,
                poly_degree=self.args.poly_degree
            ).to(self.device)

        if self.loss_fn is None:
            # Construct input bounds dictionary
            input_bounds = {
                'min': torch.tensor([-self.args.input_max]),  # Assuming symmetric bounds
                'max': torch.tensor([self.args.input_max])
            }
            
            self.loss_fn = initialize_loss(
                self.device,
                input_bounds=input_bounds,
                minWith=self.args.minWith,
                reachMode=self.args.reachMode,
                reachAim=self.args.reachAim
            )

    def train(self, counterexample: Optional[torch.Tensor] = None):
        """Train the model with optional counterexample handling."""
        self.logger.info("Initializing training")
        self.initialize_components(counterexample)
        
        # Setup data loader and continue with training
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            pin_memory=False
        )
        
        self.logger.info("Starting model training")
        train(
            model=self.model,
            train_dataloader=dataloader,
            epochs=self.args.num_epochs,
            lr=self.args.lr,
            steps_til_summary=100,
            epochs_til_checkpoint=1000,
            model_dir=self.root_path,
            loss_fn=self.loss_fn,
            clip_grad=False,
            use_lbfgs=False,
            validation_fn=self.validate,
            start_epoch=0
        )

        self.logger.info("Saving experiment details")
        save_experiment_details(self.root_path, str(self.loss_fn), vars(self.args))

    def validate(self, model, ckpt_dir, epoch):
        """Validation function called during training"""
        # Unnormalization constants
        norm_to = 0.02
        mean = 0.25
        var = 0.5

        # Define evaluation time points
        times = [self.args.tMin, 0.5 * (self.args.tMin + self.args.tMax), self.args.tMax]
        num_times = len(times)

        # Create state space sampling grid
        state_range = torch.linspace(-1.5, 1.5, 200)
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

            contour = axes[i].contourf(X, V, model_out, levels=50, cmap='bwr')
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
        self.logger.debug(f"Saved comparison plot at: {save_path}")

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
            # Use default file in the same directory as this module
            current_dir = os.path.dirname(os.path.abspath(__file__))
            matlab_file_path = os.path.join(current_dir, self.DEFAULT_MATLAB_FILE)

        self.logger.info(f"Loading true value function from: {matlab_file_path}")
        
        if not os.path.exists(matlab_file_path):
            raise FileNotFoundError(
                f"MATLAB file not found at: {matlab_file_path}. "
                f"Please ensure '{self.DEFAULT_MATLAB_FILE}' is in the same directory as double_integrator.py"
            )
            
        if self.model is None:
            raise ValueError("Neural network model not initialized")
            
        # Load MATLAB data
        matlab_data = load_matlab_data(matlab_file_path)
        
        # Add time dimension to grid points
        grid_points = matlab_data['grid']
        time_coords = torch.ones((grid_points.shape[0], 1)) * self.args.tMax
        matlab_data['grid'] = np.hstack((time_coords, grid_points))
        
        # Compare with neural network
        difference, mse = compare_with_nn(
            self.model,
            matlab_data,
            visualize=visualize
        )
        
        # Log results
        self.logger.info(f"Comparison Results:")
        self.logger.info(f"Mean Squared Error: {mse:.6f}")
        self.logger.info(f"Max Absolute Error: {np.max(np.abs(difference)):.6f}")
        self.logger.info(f"Mean Absolute Error: {np.mean(np.abs(difference)):.6f}")
        
        # Save comparison figure if visualize is True
        if visualize:
            plt.savefig(os.path.join(self.root_path, 'true_value_comparison.png'))
            plt.close()
            
        return difference, mse
