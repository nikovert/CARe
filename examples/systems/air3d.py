import logging
from typing import List, Union, Dict, Any, Callable

import torch
import matplotlib
from math import sqrt, pi
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from certreach.common.base_system import DynamicalSystem
from examples.factories import register_example

logger = logging.getLogger(__name__)

def air3d_boundary(states, collision_radius=sqrt(0.25)):
    """
    Compute boundary values for both PyTorch tensors and symbolic variables.
    
    This standalone function is designed to be serializable for multiprocessing.
    
    Args:
        states: Either a PyTorch tensor of shape [batch_size, state_dim] or a list of symbolic variables
        collision_radius: Radius for collision detection
        
    Returns:
        Boundary values in the same format as input
    """
    # Check if using PyTorch tensors
    using_torch = isinstance(states, torch.Tensor)
    
    if using_torch:
        # Extract [x, y] position (ignore theta)
        pos = states[:, :2]
        return torch.norm(pos, dim=1, keepdim=True)**2 - collision_radius**2
    else:
        # dReal/symbolic mode - extract [x, y] from states list
        if len(states) < 2:  # Need at least x and y
            return 0
        x, y = states[0], states[1]
        # Manual computation of L2 norm for symbolic variables
        return (x * x + y * y) - collision_radius**2

def air3d_hamiltonian(states, partials, func_map, velocity=0.75, omega_max=3.0, reach_aim='reach', alpha_angle=pi):
    """
    Compute the Hamiltonian for the Air3D system.
    
    This standalone function is designed to be serializable for multiprocessing.
    
    Args:
        states: List of state variables or tensor [batch_size, state_dim]
        partials: List of partial derivative variables or tensor [batch_size, state_dim]
        func_map: Dictionary mapping function names to their implementations
        velocity: Aircraft velocity
        omega_max: Maximum angular velocity
        reach_aim: 'reach' or 'avoid'
        alpha_angle: Scaling factor for theta
        
    Returns:
        Hamiltonian value in the same format as inputs
    """
    using_torch = isinstance(partials[0] if isinstance(partials, (list, tuple)) else partials[..., 0], torch.Tensor)

    ve = velocity
    vp = velocity

    if using_torch:
        # x is [batch, 3] for [x, y, theta]
        # p is [batch, 3] for [p_x, p_y, p_theta]
        pos_x, pos_y = states[..., 0], states[..., 1]
        theta = states[..., 2] * alpha_angle
        p_x, p_y, p_theta = partials[..., 0], partials[..., 1], partials[..., 2] / alpha_angle
        
        # Compute Hamiltonian terms
        ham = p_x * (-ve + vp * torch.cos(theta)) + p_y * vp * torch.sin(theta)

        # Control input term based on reach/avoid
        if reach_aim == 'avoid':
            ham += omega_max * torch.abs(p_x*pos_y - p_y*pos_x - p_theta) - omega_max * p_theta
        else:  # reach
            ham -= omega_max * torch.abs(p_x*pos_y - p_y*pos_x - p_theta) - omega_max * p_theta
    else:
        # Symbolic computation - check if we have enough variables
        if len(states) < 3 or len(partials) < 3:
            return 0
            
        pos_x, pos_y = states[0], states[1]
        theta = states[2] * alpha_angle
        p_x, p_y, p_theta = partials[0], partials[1], partials[2] / alpha_angle
        
        # Use provided function map for symbolic functions
        cos_fn = func_map.get('cos', None)
        sin_fn = func_map.get('sin', None)
        abs_fn = func_map.get('abs', abs)
        
        if cos_fn is None or sin_fn is None:
            # Fallback if functions aren't provided
            logger.warning("Missing trigonometric functions in func_map, using defaults")
            import math
            cos_fn = math.cos
            sin_fn = math.sin
            
        # Compute Hamiltonian terms for symbolic variables
        ham = p_x * (-ve + vp * cos_fn(theta)) + p_y * vp * sin_fn(theta)
        
        if reach_aim == 'avoid':
            ham += omega_max * abs_fn(p_x*pos_y - p_y*pos_x - p_theta) - omega_max * p_theta
        else:  # reach
            ham -= omega_max * abs_fn(p_x*pos_y - p_y*pos_x - p_theta) - omega_max * p_theta
            
    return ham

@register_example
class Air3D(DynamicalSystem):
    """Air3D system for reachability analysis."""
    
    Name = "air3d"
    DEFAULT_MATLAB_FILE = "data/value_function.mat"
    NUM_STATES = 3  # [x, y, theta]

    def __init__(self, args):
        super().__init__(args)
        
        # Air3D specific parameters
        self.velocity = args.velocity if hasattr(args, 'velocity') else 0.75
        self.omega_max = args.omega_max if hasattr(args, 'omega_max') else 3.0
        self.collision_radius = args.collision_radius if hasattr(args, 'collision_radius') else sqrt(0.25)
        self.alpha_angle = args.alpha_angle if hasattr(args, 'alpha_angle') else pi
        
        # Set input bounds
        self.input_bounds = {
            'min': torch.tensor([-self.omega_max]),
            'max': torch.tensor([self.omega_max])
        }
        
        # Define the boundary condition function
        self.boundary_fn = lambda states: air3d_boundary(states, self.collision_radius)
    
    def compute_hamiltonian(self, x, p, func_map: dict) -> torch.Tensor:
        """
        Instance method that delegates to the standalone function for better serializability.
        """
        return air3d_hamiltonian(
            x, p, func_map,
            velocity=self.velocity,
            omega_max=self.omega_max,
            reach_aim=self.reach_aim,
            alpha_angle=self.alpha_angle
        )

    def _get_state_names(self) -> List[str]:
        """Override state names for Air3D."""
        return ["x", "y", "θ"]