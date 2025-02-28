import logging
from typing import List

import torch
import matplotlib
from math import sqrt
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from certreach.common.base_system import DynamicalSystem
from examples.factories import register_example

logger = logging.getLogger(__name__)

def air3d_boundary(states, collision_radius=sqrt(0.25)):
    """Compute boundary values for both PyTorch tensors and dReal variables."""
    # Check if using PyTorch tensors
    using_torch = isinstance(states, torch.Tensor)
    
    if using_torch:
        # Extract [x, y] position (ignore theta)
        pos = states[:, :2]
        return torch.norm(pos, dim=1, keepdim=True)**2 - collision_radius**2
    else:
        # dReal mode - coords will be a list of variables [t, x, y, theta]
        x, y = states[0], states[1]
        # Manual computation of L2 norm for dReal
        return (x * x + y * y) - collision_radius**2

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
            if self.reach_aim == 'avoid':
                ham -= self.omega_max * torch.abs(p_theta)  # Maximize angular velocity
            else:  # reach
                ham += self.omega_max * torch.abs(p_theta)  # Minimize angular velocity
        else:
            # dReal mode - symbolic computation
            theta = x[2] * self.alpha_angle
            p_x, p_y, p_theta = p[0], p[1], p[2] / self.alpha_angle
            
            # Compute Hamiltonian terms with dReal functions
            ham = p_x * self.velocity * func_map['cos'](theta) + p_y * self.velocity * func_map['sin'](theta)
            
            # Control input term using Abs function provided
            abs_p_theta = func_map['abs'](p_theta)
            if self.reach_aim == 'avoid':
                ham -= self.omega_max * abs_p_theta  # Maximize angular velocity
            else:  # reach
                ham += self.omega_max * abs_p_theta  # Minimize angular velocity
        
        # Apply backward/forward mode adjustment
        if self.reach_mode == 'backward':
            ham = -ham
            
        return ham

    def _get_state_names(self) -> List[str]:
        """Override state names for Air3D."""
        return ["x", "y", "θ"]