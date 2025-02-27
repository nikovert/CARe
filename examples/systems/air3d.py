import os
import torch
import logging
import numpy as np
from typing import List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from certreach.common.base_system import DynamicalSystem
from examples.factories import register_example

logger = logging.getLogger(__name__)

def air3d_boundary(states, collision_radius=0.25):
    """Compute boundary values for both PyTorch tensors and dReal variables."""
    # Check if using PyTorch tensors
    using_torch = isinstance(states, torch.Tensor)
    
    if using_torch:
        # Extract [x, y] position (ignore theta)
        pos = states[:, 1:3]
        return torch.norm(pos, dim=1, keepdim=True) - collision_radius
    else:
        # dReal mode - coords will be a list of variables [t, x, y, theta]
        x, y = states[0], states[1]
        # Manual computation of L2 norm for dReal
        return (x * x + y * y)**0.5 - collision_radius

@register_example
class Air3D(DynamicalSystem):
    """Air3D system for reachability analysis."""
    
    Name = "air3d"
    DEFAULT_MATLAB_FILE = "data/value_function.mat"
    NUM_STATES = 3  # [x, y, theta]

    def __init__(self, args):
        super().__init__(args)
        
        # Air3D specific parameters
        self.velocity = args.velocity if hasattr(args, 'velocity') else 0.6
        self.omega_max = args.omega_max if hasattr(args, 'omega_max') else 1.1
        self.collision_radius = args.collision_radius if hasattr(args, 'collision_radius') else 0.25
        self.alpha_angle = args.alpha_angle if hasattr(args, 'alpha_angle') else 1.0
        
        # Set input bounds
        self.input_bounds = {
            'min': torch.tensor([-self.omega_max]),
            'max': torch.tensor([self.omega_max])
        }
        
        # Define the boundary condition function
        self.boundary_fn = lambda states: air3d_boundary(states, self.collision_radius)
    
    def compute_hamiltonian(self, x, p, func_map: dict) -> torch.Tensor:
        """Compute the Hamiltonian for the Air3D system."""
        using_torch = isinstance(p[0] if isinstance(p, (list, tuple)) else p[..., 0], torch.Tensor)

        if using_torch:
            # x is [batch, 3] for [x, y, theta]
            # p is [batch, 3] for [p_x, p_y, p_theta]
            theta = x[..., 2] * self.alpha_angle
            p_x, p_y, p_theta = p[..., 0], p[..., 1], p[..., 2] / self.alpha_angle
            
            # Compute Hamiltonian terms
            # dx/dt = v*cos(theta), dy/dt = v*sin(theta), dtheta/dt = omega
            ham = p_x * self.velocity * torch.cos(theta) + p_y * self.velocity * torch.sin(theta)
            
            # Control input term based on reach/avoid
            if self.reachAim == 'avoid':
                ham -= self.omega_max * torch.abs(p_theta)  # Maximize angular velocity
            else:  # reach
                ham += self.omega_max * torch.abs(p_theta)  # Minimize angular velocity
        else:
            # dReal mode - symbolic computation
            theta = x[2] * self.alpha_angle
            p_x, p_y, p_theta = p[0], p[1], p[2] / self.alpha_angle
            
            # Compute Hamiltonian terms with dReal functions
            ham = p_x * self.velocity * func_map['cos'](theta) + p_y * self.velocity * func_map['sin'](theta)
            
            # Control input term using Abs function provided (could be dreal.abs)
            abs_p_theta = func_map['abs'](p_theta)
            if self.reachAim == 'avoid':
                ham -= self.omega_max * abs_p_theta  # Maximize angular velocity
            else:  # reach
                ham += self.omega_max * abs_p_theta  # Minimize angular velocity
        
        # Apply backward/forward mode adjustment
        if self.reachMode == 'backward':
            ham = -ham
            
        return ham

    def _get_state_names(self) -> List[str]:
        """Override state names for Air3D."""
        return ["x", "y", "θ"]
        
    def validate(self, model, ckpt_dir, epoch):
        """Validation function for Air3D system - visualize 2D slices."""
        times = [self.args.tMin, 0.5 * (self.args.tMin + self.args.tMax), self.args.tMax]
        num_times = len(times)

        # Create state space sampling grid
        x_range = torch.linspace(-1.5, 1.5, 100)
        y_range = torch.linspace(-1.5, 1.5, 100)
        theta_slices = [-np.pi/2, 0, np.pi/2]  # 3 slices in theta dimension

        fig, axes = plt.subplots(num_times, len(theta_slices), figsize=(15, 5*num_times))

        for t_idx, t in enumerate(times):
            for theta_idx, theta in enumerate(theta_slices):
                X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
                
                # Create coordinates for this slice
                coords = torch.cat((
                    torch.ones_like(X.reshape(-1, 1)) * t,
                    X.reshape(-1, 1),
                    Y.reshape(-1, 1),
                    torch.ones_like(X.reshape(-1, 1)) * theta
                ), dim=1).to(self.device)

                model_out = model({'coords': coords})['model_out'].detach().cpu().numpy()
                model_out = model_out.reshape(X.shape)

                # Plot results
                ax = axes[t_idx, theta_idx]
                contour = ax.contourf(X.numpy(), Y.numpy(), model_out, levels=50, cmap='bwr')
                zero_level = ax.contour(X.numpy(), Y.numpy(), model_out, levels=[0], colors='k', linewidths=2)
                ax.clabel(zero_level, inline=True, fontsize=8)
                ax.set_title(f"t={t:.2f}, θ={theta:.2f}")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                fig.colorbar(contour, ax=ax)

        plt.tight_layout()
        filename = f'Air3D_val_epoch_{epoch:04d}.png'
        fig.savefig(os.path.join(ckpt_dir, filename))
        plt.close(fig)
        
    def plot_final_model(self, model, save_dir, epsilon, save_file="Air3D_Final_Model_With_Epsilon.png"):
        """Plot comparison of Air3D value functions at slices."""
        x_range = torch.linspace(-1.5, 1.5, 100)
        y_range = torch.linspace(-1.5, 1.5, 100)
        theta = 0  # Middle slice for comparison

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        fig.suptitle(f"Air3D Value Function Comparison (ε={epsilon:.4f})")

        X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
        
        # Create coordinates
        coords = torch.cat((
            torch.ones_like(X.reshape(-1, 1)) * self.args.tMax,
            X.reshape(-1, 1),
            Y.reshape(-1, 1),
            torch.ones_like(X.reshape(-1, 1)) * theta
        ), dim=1).to(self.device)

        # Get model output and adjust
        model_out = model({'coords': coords})['model_out'].cpu().detach().numpy().reshape(X.shape)
        adjusted_model_out = model_out - epsilon

        # Original value function
        contour1 = axes[0].contourf(X.numpy(), Y.numpy(), model_out, levels=50, cmap='bwr')
        zero_level1 = axes[0].contour(X.numpy(), Y.numpy(), model_out, levels=[0], colors='k', linewidths=2)
        axes[0].clabel(zero_level1, inline=True, fontsize=8)
        axes[0].set_title("Original Value Function")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(contour1, ax=axes[0])

        # Epsilon-adjusted value function
        contour2 = axes[1].contourf(X.numpy(), Y.numpy(), adjusted_model_out, levels=50, cmap='bwr')
        zero_level2 = axes[1].contour(X.numpy(), Y.numpy(), adjusted_model_out, levels=[0], colors='k', linewidths=2)
        axes[1].clabel(zero_level2, inline=True, fontsize=8)
        axes[1].set_title(f"Epsilon-Adjusted Value")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        fig.colorbar(contour2, ax=axes[1])

        # Zero-level set comparison
        axes[2].contour(X.numpy(), Y.numpy(), model_out, levels=[0], colors='b', linewidths=2, linestyles='--')
        axes[2].contour(X.numpy(), Y.numpy(), adjusted_model_out, levels=[0], colors='r', linewidths=2)
        axes[2].set_title("Zero-Level Set Comparison")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].legend(["Original", f"ε-Adjusted"], loc="upper right")
        axes[2].grid(True)

        plt.tight_layout()
        save_path = os.path.join(save_dir, save_file)
        plt.savefig(save_path)
        plt.close(fig)
