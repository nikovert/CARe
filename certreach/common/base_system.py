import os
import torch
import logging
import inspect
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Callable, List, Dict, Union, Tuple

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
        self.device = torch.device(getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.root_path = None  # Set by experiment manager
        
        # Model and functions are initialized later
        self.model = None
        self.loss_fn = None
        self.hamiltonian_fn = None
        self.boundary_fn = None  # Must be implemented by subclass
        
        # Initialize input bounds (subclass should set these)
        self.input_bounds = None
        
        # Loss settings from args
        self.min_with = args.min_with if hasattr(args, 'min_with') else 'none'
        self.reach_mode = args.reach_mode if hasattr(args, 'reach_mode') else 'backward'
        self.reach_aim = args.reach_aim if hasattr(args, 'reach_aim') else 'reach'
        
    def initialize_components(self):
        """
        Initialize model, loss function, and hamiltonian.
        Should be called before training or verification.
        """
        # Initialize model if needed
        if self.model is None:
            config = NetworkConfig(
                in_features=self.NUM_STATES+1,
                out_features=1,
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
            min_with=self.min_with, 
            reach_mode=self.reach_mode, 
            reach_aim=self.reach_aim
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
            t_min=self.args.t_min,
            t_max=self.args.t_max,
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
            time_min=self.args.t_min,
            time_max=self.args.t_max,
            validation_fn=self.validate,
            device=self.device,
            use_amp=True
        )
    
    def _get_state_names(self) -> List[str]:
        """Get state variable names for plotting."""
        return [f"x{i+1}" for i in range(self.NUM_STATES)]
    
    def validate(self, model, ckpt_dir, epoch, t_max=None):
        """
        Validation function called during training.
        Visualizes value function at different time slices.
        
        For 1D systems: Plots the value function directly
        For 2D systems: Creates contour plots
        For 3D+ systems: Creates contour plots of 2D slices with fixed values for other dimensions
        """
        # Define evaluation time points
        if t_max is None or t_max == self.args.t_max:
            times = [self.args.t_min, 0.5*(self.args.t_min + self.args.t_max), self.args.t_max]
        else:
            times = [self.args.t_min, t_max, self.args.t_max]
        
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
        
        else:
            # Higher-dimensional state spaces (n > 2)
            # For systems with more than 2 states, we'll visualize 2D slices
            # by fixing the values of the remaining states
            
            # Fixed values for states beyond the first two
            # Default: all fixed values are 0.0
            fixed_states = getattr(self, 'validation_fixed_states', [0.0] * (self.NUM_STATES - 2))
            
            # Number of slices to show (can be customized by subclasses)
            num_slices = getattr(self, 'validation_num_slices', 1)
            slice_values = []
            
            if hasattr(self, 'validation_slice_values'):
                # Use predefined slice values if provided
                slice_values = self.validation_slice_values
                num_slices = len(slice_values)
            else:
                # Use the first state beyond x1,x2 for slicing, with predefined values
                slice_dim = getattr(self, 'validation_slice_dim', 2)  # Default: the 3rd state (index 2)
                slice_values = []
                
                # Default slices: centered at 0 with equal spacing
                if num_slices == 1:
                    slice_values = [0.0]
                else:
                    slice_values = np.linspace(-0.5, 0.5, num_slices)
            
            # Create a figure with a grid of plots: times × slices
            fig, axes = plt.subplots(num_times, num_slices, 
                                     figsize=(5 * num_slices, 5 * num_times))
            
            # Handle case where we have only one time or slice
            if num_times == 1 and num_slices == 1:
                axes = np.array([[axes]])
            elif num_times == 1:
                axes = axes[np.newaxis, :]
            elif num_slices == 1:
                axes = axes[:, np.newaxis]
            
            # For each time and slice, create a 2D contour plot
            for i, t in enumerate(times):
                for j, slice_val in enumerate(slice_values):
                    # Create a meshgrid for the first two states
                    X1, X2 = torch.meshgrid(state_range, state_range, indexing='ij')
                    x1_flat = X1.reshape(-1, 1)
                    x2_flat = X2.reshape(-1, 1)
                    
                    # Create constant values for time and remaining states
                    batch_size = x1_flat.shape[0]
                    time_coords = torch.ones(batch_size, 1) * t
                    
                    # Create coordinates for all states
                    state_tensors = [x1_flat, x2_flat]
                    
                    # If we have specific slice values, update fixed_states accordingly
                    current_fixed_states = fixed_states.copy()
                    if hasattr(self, 'validation_slice_dim'):
                        slice_dim = self.validation_slice_dim
                        if 0 <= slice_dim - 2 < len(current_fixed_states):
                            current_fixed_states[slice_dim - 2] = slice_val
                    
                    # Add the fixed values for remaining states
                    for fixed_val in current_fixed_states:
                        state_tensors.append(torch.ones(batch_size, 1) * fixed_val)
                    
                    # Concatenate all states with time
                    coords = torch.cat([time_coords] + state_tensors, dim=1).to(self.device)
                    
                    # Get model output
                    model_in = {'coords': coords}
                    model_out = model(model_in)['model_out'].detach().cpu().numpy().reshape(X1.shape)
                    
                    # Create the contour plot
                    ax = axes[i, j]
                    contour = ax.contourf(X1, X2, model_out, levels=50, cmap='bwr')
                    
                    # Add zero level set contour
                    zero_contour = ax.contour(X1, X2, model_out, levels=[0], colors='k', linewidths=2)
                    ax.clabel(zero_contour, inline=True, fontsize=8)
                    
                    # Generate title based on which dimension is varied
                    if hasattr(self, 'validation_slice_dim'):
                        slice_name = state_names[self.validation_slice_dim]
                        ax.set_title(f"t={t:.2f}, {slice_name}={slice_val:.2f}")
                    else:
                        # Create a title that shows all fixed values
                        fixed_vals_str = ", ".join([f"{state_names[i+2]}={val:.2f}" 
                                                   for i, val in enumerate(current_fixed_states)])
                        ax.set_title(f"t={t:.2f}, {fixed_vals_str}")
                    
                    ax.set_xlabel(state_names[0])
                    ax.set_ylabel(state_names[1])
                    fig.colorbar(contour, ax=ax)
            
        plt.tight_layout()
        filename = f'{self.Name}_val_epoch_{epoch:04d}.png'
        fig.savefig(os.path.join(ckpt_dir, filename))
        plt.close(fig)

    def plot_final_model(self, model, save_dir, epsilon, save_file=None):
        """
        Plot final model results with epsilon adjustment.
        
        For 1D systems: Plots the value function directly
        For 2D systems: Creates contour plots
        For 3D+ systems: Creates contour plots of 2D slices with fixed values for other dimensions
        
        Args:
            model: Trained model
            save_dir: Directory to save plot
            epsilon: Epsilon value for robustness adjustment
            save_file: File name for saved plot
        """
        if save_file is None:
            save_file = f"{self.Name}_Final_Model_Comparison_With_Zero_Set.png"
        
        # Create state space sampling grid
        state_range = torch.linspace(-1, 1, 200)
        state_names = self._get_state_names()
        
        if self.NUM_STATES == 1:
            # 1D state space
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            X = state_range
            time_coords = torch.ones_like(X.reshape(-1, 1)) * self.args.t_max
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
            time_coords = torch.ones_like(x1_flat) * self.args.t_max
            
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
            
        else:
            # Higher-dimensional state spaces (n > 2)
            # For systems with more than 2 states, we'll visualize 2D slices
            
            # Fixed values for states beyond the first two
            # Default: all fixed values are 0.0
            fixed_states = getattr(self, 'validation_fixed_states', [0.0] * (self.NUM_STATES - 2))
            
            # Use middle slice for final model plot
            slice_val = 0.0
            if hasattr(self, 'validation_slice_values') and len(self.validation_slice_values) > 0:
                # Use middle slice from predefined values if available
                middle_idx = len(self.validation_slice_values) // 2
                slice_val = self.validation_slice_values[middle_idx]
            
            # Create the figure
            fig, axes = plt.subplots(1, 3, figsize=(21, 6))
            fig.suptitle(f"Value Function (ε={epsilon:.4f})")
            
            # Create a meshgrid for the first two states
            X1, X2 = torch.meshgrid(state_range, state_range, indexing='ij')
            x1_flat = X1.reshape(-1, 1)
            x2_flat = X2.reshape(-1, 1)
            
            # Create constant values for time and remaining states
            batch_size = x1_flat.shape[0]
            time_coords = torch.ones(batch_size, 1) * self.args.t_max
            
            # Create coordinates for all states
            state_tensors = [x1_flat, x2_flat]
            
            # If we have specific slice values, update fixed_states accordingly
            current_fixed_states = fixed_states.copy()
            if hasattr(self, 'validation_slice_dim'):
                slice_dim = self.validation_slice_dim
                if 0 <= slice_dim - 2 < len(current_fixed_states):
                    current_fixed_states[slice_dim - 2] = slice_val
            
            # Add the fixed values for remaining states
            for fixed_val in current_fixed_states:
                state_tensors.append(torch.ones(batch_size, 1) * fixed_val)
            
            # Concatenate all states with time
            coords = torch.cat([time_coords] + state_tensors, dim=1).to(self.device)
            
            # Get model output
            model_in = {'coords': coords}
            model_out = model(model_in)['model_out'].detach().cpu().numpy().reshape(X1.shape)
            adjusted_model_out = model_out - epsilon
            
            # Generate title suffix showing fixed states
            if hasattr(self, 'validation_slice_dim'):
                slice_name = state_names[self.validation_slice_dim]
                fixed_states_str = f"{slice_name}={slice_val:.2f}"
            else:
                fixed_states_str = ", ".join([f"{state_names[i+2]}={val:.2f}" 
                                           for i, val in enumerate(current_fixed_states)])
            
            # Original Value Function
            contour1 = axes[0].contourf(X1, X2, model_out, levels=50, cmap='bwr')
            zero_level1 = axes[0].contour(X1, X2, model_out, levels=[0], colors='k', linewidths=2)
            axes[0].clabel(zero_level1, inline=True, fontsize=8)
            axes[0].set_title(f"Original Value ({fixed_states_str})")
            axes[0].set_xlabel(state_names[0])
            axes[0].set_ylabel(state_names[1])
            fig.colorbar(contour1, ax=axes[0])
            
            # Epsilon-Adjusted Value Function
            contour2 = axes[1].contourf(X1, X2, adjusted_model_out, levels=50, cmap='bwr')
            zero_level2 = axes[1].contour(X1, X2, adjusted_model_out, levels=[0], colors='k', linewidths=2)
            axes[1].clabel(zero_level2, inline=True, fontsize=8)
            axes[1].set_title(f"Epsilon-Adjusted Value ({fixed_states_str})")
            axes[1].set_xlabel(state_names[0])
            axes[1].set_ylabel(state_names[1])
            fig.colorbar(contour2, ax=axes[1])
            
            # Zero-Level Set Comparison
            axes[2].contour(X1, X2, model_out, levels=[0], colors='b', linewidths=2, linestyles='--')
            axes[2].contour(X1, X2, adjusted_model_out, levels=[0], colors='r', linewidths=2, linestyles='-')
            axes[2].set_title(f"Zero-Level Set Comparison ({fixed_states_str})")
            axes[2].set_xlabel(state_names[0])
            axes[2].set_ylabel(state_names[1])
            axes[2].legend(["Original", f"Epsilon-Adjusted"], loc="upper right")
            axes[2].grid(True)
            
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
