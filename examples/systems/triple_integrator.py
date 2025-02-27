import os
import torch
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List

from certreach.common.base_system import DynamicalSystem
from examples.factories import register_example

logger = logging.getLogger(__name__)

def triple_integrator_boundary(states, radius=0.25):
    """Compute boundary values for both PyTorch tensors and dReal variables."""
    # Check if using PyTorch tensors
    using_torch = isinstance(states, torch.Tensor)
    
    if using_torch:
        # Extract position states (first 3 dimensions)
        if states.dim() > 1 and states.size(1) > 3:
            pos = states[:, 0:3]  # Extract position states
        else:
            pos = states  # Assume all states are positions
            
        return torch.norm(pos, dim=1, keepdim=True) - radius
    else:
        # dReal mode - states will be a list of variables [x1, x2, x3]
        x1, x2, x3 = states[0], states[1], states[2]
        # Manual computation of L2 norm for dReal
        norm = (x1 * x1 + x2 * x2 + x3 * x3) ** 0.5
        return norm - radius

@register_example
class TripleIntegrator(DynamicalSystem):
    """Triple Integrator system for reachability analysis."""
    
    Name = "triple_integrator"
    DEFAULT_MATLAB_FILE = "data/value_function.mat"  # Update if you have a MATLAB file
    NUM_STATES = 3  # position, velocity, acceleration
    
    def __init__(self, args):
        super().__init__(args)
        
        # Initialize input bounds specific to Triple Integrator
        self.input_bounds = {
            'min': torch.tensor([-args.input_max]),
            'max': torch.tensor([args.input_max])
        }
        
        # Define the boundary condition function
        self.boundary_fn = triple_integrator_boundary
    
    def compute_hamiltonian(self, x, p, func_map: dict) -> torch.Tensor:
        """
        Compute the Hamiltonian for the Triple Integrator system.
        
        Args:
            x: State variables [x1, x2, x3] (position, velocity, acceleration)
            p: Costate variables [p1, p2, p3]
            Abs: Function to compute absolute value (for SMT call compatibility)
            
        Returns:
            Hamiltonian value
        """
        using_torch = isinstance(p[0] if isinstance(p, (list, tuple)) else p[..., 0], torch.Tensor)
        
        if using_torch:
            p1, p2, p3 = p[..., 0], p[..., 1], p[..., 2]
            x2, x3 = x[..., 1], x[..., 2]
        else:
            p1, p2, p3 = p[0], p[1], p[2]
            x2, x3 = x[1], x[2]
            
        # Dynamics: ẋ1 = x2, ẋ2 = x3, ẋ3 = u
        ham = p1 * x2 + p2 * x3
        
        # Add control contribution
        input_max = self.input_bounds['max'].to(self.device) if using_torch else float(self.input_bounds['max'].item())
        
        if self.reach_aim == 'reach':
            if using_torch:
                ham += input_max * torch.abs(p3)
            else:
                ham += float(input_max) * func_map['abs'](p3)
        else:  # avoid
            if using_torch:
                ham -= input_max * torch.abs(p3)
            else:
                ham -= float(input_max) * func_map['abs'](p3)
                
        return ham

    def _get_state_names(self) -> List[str]:
        """Override state names for Triple Integrator."""
        return ["Position (x)", "Velocity (v)", "Acceleration (a)"]
        
    def validate(self, model, ckpt_dir, epoch, tmax=None):
        """Validation function for triple integrator - we need 2D slice visualization."""
        state_range = torch.linspace(-1.5, 1.5, 50)
        times = [self.args.tMin, 0.5 * (self.args.tMin + self.args.tMax), self.args.tMax]
        num_times = len(times)

        fig = plt.figure(figsize=(15, 5 * num_times))
        grid = plt.GridSpec(num_times, 3, figure=fig)

        for t_idx, t in enumerate(times):
            X1, X2, X3 = torch.meshgrid(state_range, state_range, state_range, indexing='ij')
            coords = torch.cat((
                torch.ones_like(X1.reshape(-1, 1)) * t,
                X1.reshape(-1, 1),
                X2.reshape(-1, 1),
                X3.reshape(-1, 1)
            ), dim=1).to(self.device)

            model_in = {'coords': coords}
            model_out = model(model_in)['model_out'].detach().cpu().numpy()
            model_out = model_out.reshape(X1.shape)

            slices = [(0, 1, 25), (1, 2, 25), (0, 2, 25)]
            titles = ['Position-Velocity', 'Velocity-Acceleration', 'Position-Acceleration']
            
            for idx, (i, j, k) in enumerate(slices):
                ax = fig.add_subplot(grid[t_idx, idx])
                slice_data = model_out[:, :, k]
                contour = ax.contourf(state_range, state_range, slice_data, levels=50, cmap='bwr')
                ax.contour(state_range, state_range, slice_data, levels=[0], colors='k', linewidths=2)
                ax.set_title(f"{titles[idx]} at t={t:.2f}")
                ax.set_xlabel(self._get_state_names()[i])
                ax.set_ylabel(self._get_state_names()[j])
                fig.colorbar(contour, ax=ax)

        plt.tight_layout()
        fig.savefig(os.path.join(ckpt_dir, f'TripleIntegrator_val_epoch_{epoch:04d}.png'))
        plt.close(fig)

    def plot_final_model(self, model, save_dir, epsilon, save_file="Triple_Integrator_Final_Model_Comparison.png"):
        """Plot comparison of triple integrator value functions with slices."""
        state_range = torch.linspace(-1, 1, 50)
        slice_pos = 25

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Triple Integrator Value Function Comparison (ε={epsilon})")

        X1, X2, X3 = torch.meshgrid(state_range, state_range, state_range, indexing='ij')
        coords = torch.cat((
            torch.ones_like(X1.reshape(-1, 1)) * self.args.tMax,
            X1.reshape(-1, 1),
            X2.reshape(-1, 1),
            X3.reshape(-1, 1)
        ), dim=1).to(self.device)

        model_in = {'coords': coords}
        model_out = model(model_in)['model_out'].cpu().detach().numpy().reshape(X1.shape)
        adjusted_model_out = model_out - epsilon

        slices = [(0, 1), (1, 2), (0, 2)]
        titles = ['Position-Velocity', 'Velocity-Acceleration', 'Position-Acceleration']

        for idx, (i, j) in enumerate(slices):
            # Original
            if i == 0 and j == 1:
                slice_data = model_out[:, :, slice_pos]
            elif i == 1 and j == 2:
                slice_data = model_out[:, slice_pos, :]
            else:  # i == 0 and j == 2
                slice_data = model_out[:, :, slice_pos].transpose()
                
            contour1 = axes[0, idx].contourf(state_range, state_range, slice_data, levels=50, cmap='bwr')
            axes[0, idx].contour(state_range, state_range, slice_data, levels=[0], colors='k', linewidths=2)
            axes[0, idx].set_title(f"Original: {titles[idx]}")
            axes[0, idx].set_xlabel(self._get_state_names()[i])
            axes[0, idx].set_ylabel(self._get_state_names()[j])
            fig.colorbar(contour1, ax=axes[0, idx])

            # Epsilon-adjusted
            if i == 0 and j == 1:
                slice_data = adjusted_model_out[:, :, slice_pos]
            elif i == 1 and j == 2:
                slice_data = adjusted_model_out[:, slice_pos, :]
            else:  # i == 0 and j == 2
                slice_data = adjusted_model_out[:, :, slice_pos].transpose()
                
            contour2 = axes[1, idx].contourf(state_range, state_range, slice_data, levels=50, cmap='bwr')
            axes[1, idx].contour(state_range, state_range, slice_data, levels=[0], colors='k', linewidths=2)
            axes[1, idx].set_title(f"ε-Adjusted: {titles[idx]}")
            axes[1, idx].set_xlabel(self._get_state_names()[i])
            axes[1, idx].set_ylabel(self._get_state_names()[j])
            fig.colorbar(contour2, ax=axes[1, idx])

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, save_file))
        plt.close(fig)
