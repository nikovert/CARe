import os
import torch
import logging
import inspect
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Callable, List

from certreach.learning.networks import SingleBVPNet, NetworkConfig
from certreach.common.dataset import ReachabilityDataset
from certreach.learning.training import train
from certreach.common.matlab_loader import load_matlab_data, compare_with_nn
from certreach.learning.loss_functions import HJILossFunction

logger = logging.getLogger(__name__)

class DynamicalSystem:
    """Base class for all dynamical systems used in reachability analysis."""
    
    Name = "base_system"  # Override in subclass
    NUM_STATES = None     # Override in subclass
    DEFAULT_MATLAB_FILE = None  # Override in subclass if using MATLAB comparison
    
    def __init__(self, args):
        """
        Initialize the dynamical system.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.device = torch.device(args.device)
        self.root_path = None  # Set by experiment manager
        
        # Model and functions are initialized later
        self.model = None
        self.loss_fn = None
        self.hamiltonian_fn = None
        self.boundary_fn = None  # Must be implemented by subclass
        
        # Initialize input bounds (subclass should set these)
        self.input_bounds = None
        
        # Loss settings from args
        self.minWith = args.minWith if hasattr(args, 'minWith') else 'none'
        self.reachMode = args.reachMode if hasattr(args, 'reachMode') else 'backward'
        self.reachAim = args.reachAim if hasattr(args, 'reachAim') else 'reach'
        
    def initialize_components(self):
        """
        Initialize model, loss function, and hamiltonian.
        Should be called before training or verification.
        """
        # Initialize model if needed
        if self.model is None:
            config = NetworkConfig(
                in_features=self.args.in_features,
                out_features=self.args.out_features,
                hidden_features=self.args.num_nl,
                num_hidden_layers=self.args.num_hl,
                activation_type=self.args.model_type, 
                use_polynomial=self.args.use_polynomial,
                poly_degree=self.args.poly_degree
            )
            
            self.model = SingleBVPNet(config=config).to(self.device)
        
        # Initialize loss and hamiltonian functions
        if self.loss_fn is None:
            self.loss_fn = self._create_loss_function()
            
        if self.hamiltonian_fn is None:
            self.hamiltonian_fn=self.compute_hamiltonian
    
    def _create_loss_function(self):
        """Create a loss function using the system's Hamiltonian."""
        # Create the HJI Loss Function passing our system's Hamiltonian
        loss_function = HJILossFunction(
            hamiltonian_fn=self.compute_hamiltonian,
            minWith=self.minWith, 
            reachMode=self.reachMode, 
            reachAim=self.reachAim
        )
        
        return loss_function.compute_loss
    
    def compute_hamiltonian(self, x, p, Abs: Callable = abs) -> torch.Tensor:
        """
        Compute the Hamiltonian for the dynamical system.
        Must be implemented by subclasses.
        
        Args:
            x: State variables
            p: Costate variables
            Abs: Function to compute absolute value (for SMT call compatibility)
            
        Returns:
            Hamiltonian value
        """
        raise NotImplementedError("Subclasses must implement compute_hamiltonian")
    
    def train(self):
        """Train the model on the dynamical system."""
        self.initialize_components()
        
        if self.NUM_STATES is None:
            raise ValueError("NUM_STATES must be defined in the subclass")
        
        dataset = ReachabilityDataset(
            batch_size=self.args.batch_size,
            tMin=self.args.tMin,
            tMax=self.args.tMax,
            seed=self.args.seed,
            device=self.device,
            num_states=self.NUM_STATES,
            compute_boundary_values=self.boundary_fn,
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
            use_amp=True
        )
    
    def _get_state_names(self) -> List[str]:
        """Get state variable names for plotting."""
        return [f"x{i+1}" for i in range(self.NUM_STATES)]
    
    def validate(self, model, ckpt_dir, epoch, tmax=None):
        """
        Validation function called during training.
        Visualizes value function at different time slices.
        
        Currently supports visualization for 1D and 2D state spaces.
        For higher-dimensional systems, subclasses should override this method.
        """
        if self.NUM_STATES > 2:
            logger.warning("Default validate() only supports 1D and 2D state spaces. "
                          f"This system has {self.NUM_STATES} states. "
                          "Consider implementing a custom validate() method.")
            return
        
        # Define evaluation time points
        if tmax is None or tmax == self.args.tMax:
            times = [self.args.tMin, 0.5*(self.args.tMin + self.args.tMax), self.args.tMax]
        else:
            times = [self.args.tMin, tmax, self.args.tMax]
        
        num_times = len(times)
        state_names = self._get_state_names()
        
        # Create state space sampling grid
        state_range = torch.linspace(-1, 1, 200)
        
        if self.NUM_STATES == 1:
            # 1D state space
            fig, axes = plt.subplots(num_times, 1, figsize=(10, 5 * num_times))
            if num_times == 1:
                axes = [axes]  # Make it iterable
                
            for i, t in enumerate(times):
                X = state_range
                time_coords = torch.ones_like(X.reshape(-1, 1)) * t
                coords = torch.cat((time_coords, X.reshape(-1, 1)), dim=1).to(self.device)
                
                model_in = {'coords': coords}
                model_out = model(model_in)['model_out'].detach().cpu().numpy()
                
                axes[i].plot(X.numpy(), model_out, 'b-', linewidth=2)
                axes[i].plot(X.numpy(), np.zeros_like(X.numpy()), 'k--', alpha=0.5)
                axes[i].set_title(f"t = {t:.2f}")
                axes[i].set_xlabel(state_names[0])
                axes[i].set_ylabel("Value")
                axes[i].grid(True)
                
        elif self.NUM_STATES == 2:
            # 2D state space
            fig, axes = plt.subplots(num_times, 1, figsize=(10, 5 * num_times))
            if num_times == 1:
                axes = [axes]  # Make it iterable
                
            for i, t in enumerate(times):
                X1, X2 = torch.meshgrid(state_range, state_range, indexing='ij')
                x1_flat = X1.reshape(-1, 1)
                x2_flat = X2.reshape(-1, 1)
                time_coords = torch.ones_like(x1_flat) * t
                
                coords = torch.cat((time_coords, x1_flat, x2_flat), dim=1).to(self.device)
                model_in = {'coords': coords}
                model_out = model(model_in)['model_out'].detach().cpu().numpy().reshape(X1.shape)
                
                # Create filled contour plot
                contour = axes[i].contourf(X1, X2, model_out, levels=50, cmap='bwr')
                
                # Add zero level set
                zero_contour = axes[i].contour(X1, X2, model_out, levels=[0], colors='k', linewidths=2)
                axes[i].clabel(zero_contour, inline=True, fontsize=8)
                
                axes[i].set_title(f"t = {t:.2f}")
                axes[i].set_xlabel(state_names[0])
                axes[i].set_ylabel(state_names[1])
                fig.colorbar(contour, ax=axes[i])
        
        plt.tight_layout()
        filename = f'{self.Name}_val_epoch_{epoch:04d}.png'
        fig.savefig(os.path.join(ckpt_dir, filename))
        plt.close(fig)
    
    def plot_final_model(self, model, save_dir, epsilon, save_file=None):
        """
        Plot final model results with epsilon adjustment.
        
        Currently supports visualization for 1D and 2D state spaces.
        For higher-dimensional systems, subclasses should override this method.
        
        Args:
            model: Trained model
            save_dir: Directory to save plot
            epsilon: Epsilon value for robustness adjustment
            save_file: File name for saved plot
        """
        if self.NUM_STATES > 2:
            logger.warning("Default plot_final_model() only supports 1D and 2D state spaces. "
                          f"This system has {self.NUM_STATES} states. "
                          "Consider implementing a custom plot_final_model() method.")
            return
            
        if save_file is None:
            save_file = f"{self.Name}_Final_Model_Comparison_With_Zero_Set.png"
        
        # Create state space sampling grid
        state_range = torch.linspace(-1, 1, 200)
        state_names = self._get_state_names()
        
        if self.NUM_STATES == 1:
            # 1D state space
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            X = state_range
            time_coords = torch.ones_like(X.reshape(-1, 1)) * self.args.tMax
            coords = torch.cat((time_coords, X.reshape(-1, 1)), dim=1).to(self.device)
            
            model_in = {'coords': coords}
            model_out = model(model_in)['model_out'].cpu().detach().numpy()
            adjusted_model_out = model_out - epsilon
            
            # Original Value Function
            axes[0].plot(X.numpy(), model_out, 'b-', linewidth=2, label="Original")
            axes[0].plot(X.numpy(), np.zeros_like(X.numpy()), 'k--', alpha=0.5)
            axes[0].set_title("Original Value Function")
            axes[0].set_xlabel(state_names[0])
            axes[0].set_ylabel("Value")
            axes[0].grid(True)
            
            # Epsilon-Adjusted Value Function
            axes[1].plot(X.numpy(), model_out, 'b--', linewidth=2, label="Original")
            axes[1].plot(X.numpy(), adjusted_model_out, 'r-', linewidth=2, label=f"Epsilon-Adjusted (ε={epsilon})")
            axes[1].plot(X.numpy(), np.zeros_like(X.numpy()), 'k--', alpha=0.5)
            axes[1].set_title("Value Function Comparison")
            axes[1].set_xlabel(state_names[0])
            axes[1].set_ylabel("Value")
            axes[1].legend()
            axes[1].grid(True)
            
        elif self.NUM_STATES == 2:
            # 2D state space
            fig, axes = plt.subplots(1, 3, figsize=(21, 6))
            
            X1, X2 = torch.meshgrid(state_range, state_range, indexing='ij')
            x1_flat = X1.reshape(-1, 1)
            x2_flat = X2.reshape(-1, 1)
            time_coords = torch.ones_like(x1_flat) * self.args.tMax
            
            coords = torch.cat((time_coords, x1_flat, x2_flat), dim=1).to(self.device)
            model_in = {'coords': coords}
            model_out = model(model_in)['model_out'].cpu().detach().numpy().reshape(X1.shape)
            adjusted_model_out = model_out - epsilon
            
            # Original Value Function
            contour1 = axes[0].contourf(X1, X2, model_out, levels=50, cmap='bwr')
            zero_level1 = axes[0].contour(X1, X2, model_out, levels=[0], colors='k', linewidths=2)
            axes[0].clabel(zero_level1, inline=True, fontsize=8)
            axes[0].set_title("Original Value Function")
            axes[0].set_xlabel(state_names[0])
            axes[0].set_ylabel(state_names[1])
            fig.colorbar(contour1, ax=axes[0])
            
            # Epsilon-Adjusted Value Function
            contour2 = axes[1].contourf(X1, X2, adjusted_model_out, levels=50, cmap='bwr')
            zero_level2 = axes[1].contour(X1, X2, adjusted_model_out, levels=[0], colors='k', linewidths=2)
            axes[1].clabel(zero_level2, inline=True, fontsize=8)
            axes[1].set_title(f"Epsilon-Adjusted Value ($\epsilon$={epsilon})")
            axes[1].set_xlabel(state_names[0])
            axes[1].set_ylabel(state_names[1])
            fig.colorbar(contour2, ax=axes[1])
            
            # Zero-Level Set Comparison
            axes[2].contour(X1, X2, model_out, levels=[0], colors='b', linewidths=2, linestyles='--')
            axes[2].contour(X1, X2, adjusted_model_out, levels=[0], colors='r', linewidths=2, linestyles='-')
            axes[2].set_title("Zero-Level Set Comparison")
            axes[2].set_xlabel(state_names[0])
            axes[2].set_ylabel(state_names[1])
            axes[2].legend(["Original", f"Epsilon-Adjusted ($\epsilon$={epsilon})"], loc="upper right")
            
        plt.tight_layout()
        save_path = os.path.join(save_dir, save_file)
        plt.savefig(save_path)
        plt.close(fig)
        logger.debug(f"Saved comparison plot at: {save_path}")
    
    def compare_with_true_values(self, matlab_file_path: Optional[str] = None, visualize: bool = True):
        """
        Compare neural network predictions with true value function from MATLAB.
        
        Args:
            matlab_file_path: Optional path to the .mat file. If None, uses default file
            visualize: Whether to plot the comparison
            
        Returns:
            Tuple of (difference array, mean squared error)
        """
        if self.DEFAULT_MATLAB_FILE is None:
            logger.warning("DEFAULT_MATLAB_FILE not defined in subclass, cannot compare with true values")
            return None, None
            
        if matlab_file_path is None:
            # Find the module where the system class is defined
            system_module = inspect.getmodule(self.__class__)
            if system_module:
                # Get the directory where the system module is located
                system_dir = os.path.dirname(os.path.abspath(system_module.__file__))
                matlab_file_path = os.path.join(system_dir, self.DEFAULT_MATLAB_FILE)
            else:
                # Fallback to current directory if module can't be determined
                current_dir = os.path.dirname(os.path.abspath(__file__))
                matlab_file_path = os.path.join(current_dir, self.DEFAULT_MATLAB_FILE)

        logger.debug("Loading true value function from: %s", matlab_file_path)
        
        if not os.path.exists(matlab_file_path):
            logger.error("MATLAB file not found: %s", matlab_file_path)
            raise FileNotFoundError(
                f"MATLAB file not found at: {matlab_file_path}. "
                f"Please ensure '{self.DEFAULT_MATLAB_FILE}' exists in the appropriate directory."
            )
            
        if self.model is None:
            raise ValueError("Neural network model not initialized")
            
        # Load MATLAB data
        matlab_data = load_matlab_data(matlab_file_path)
        
        # Compare with neural network
        save_path = os.path.join(self.root_path, f'{self.Name}_true_value_comparison.png')
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
