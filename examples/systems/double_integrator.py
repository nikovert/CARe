import torch
import logging
from math import sqrt
from typing import List, Union, Dict, Any, Callable

from certreach.common.base_system import DynamicalSystem
from examples.factories import register_example


logger = logging.getLogger(__name__)

def double_integrator_boundary(states, radius=sqrt(0.25)):
    """
    Compute boundary values for both PyTorch tensors and dReal/symbolic variables.
    
    This function is designed to be serializable for multiprocessing.
    
    Args:
        states: Either a PyTorch tensor of shape [batch_size, state_dim] or a list of symbolic variables
        radius: Collision radius
        
    Returns:
        Boundary values in the same format as input
    """
    # Check if using PyTorch tensors
    using_torch = isinstance(states, torch.Tensor)
    
    if using_torch:
        return torch.norm(states, dim=1, keepdim=True)**2 - radius**2
    else:
        # Symbolic computation - unpack states
        # States length can vary based on the system dimension
        if not states:
            return 0
            
        sum_squares = sum(s*s for s in states)
        return sum_squares - radius**2

def double_integrator_hamiltonian(states, partials, func_map, input_bounds, reach_aim='reach'):
    """
    Compute the Hamiltonian for the Double Integrator system.
    
    This standalone function is designed to be serializable for multiprocessing.
    
    Args:
        states: List of state variables or tensor [batch_size, state_dim]
        partials: List of partial derivative variables or tensor [batch_size, state_dim]
        func_map: Dictionary mapping function names to their implementations
        input_bounds: Optional dictionary with 'min' and 'max' control bounds
        reach_aim: 'reach' or 'avoid'
        
    Returns:
        Hamiltonian value in the same format as inputs
    """
    using_torch = isinstance(partials[0] if isinstance(partials, (list, tuple)) else partials[..., 0], torch.Tensor)
    
    if using_torch:
        p1, p2 = partials[..., 0], partials[..., 1]
        x2 = states[..., 1]
    else:
        # For symbolic variables (list format)
        if len(partials) < 2 or len(states) < 2:
            return 0
        p1, p2 = partials[0], partials[1]
        x2 = states[1]
        
    # Basic part of Hamiltonian 
    ham = p1 * x2

    input_min = input_bounds['min']
    input_max = input_bounds['max']
    is_symmetric = torch.equal(input_max, -input_min)
    
    if is_symmetric:
        input_magnitude = input_max
        sign = 1 if reach_aim == 'avoid' else -1
        
        if using_torch:
            ham += sign * input_magnitude * torch.abs(p2)
        else:
            abs_p2 = func_map['abs'](p2) if 'abs' in func_map else abs(p2)
            ham += float(sign * input_magnitude) * abs_p2
    else:
        # Asymmetric bounds branch with arithmetic formulation
        if using_torch:
            if reach_aim == 'reach':
                ham += torch.where(p2 >= 0, input_min * p2, input_max * p2)
            else:  # reach
                ham += torch.where(p2 >= 0, input_max * p2, input_min * p2)
        else:
            # For symbolic computation
            abs_p2 = func_map['abs'](p2) if 'abs' in func_map else abs(p2)
            if reach_aim == 'avoid':
                ham += (float((input_max + input_min)/2) * p2 + float((input_max - input_min)/2) * abs_p2)
            else:  # avoid
                ham += (float((input_max + input_min)/2) * p2 - float((input_max - input_min)/2) * abs_p2)
    
    return ham

@register_example
class DoubleIntegrator(DynamicalSystem):
    """Double Integrator system for reachability analysis."""
    
    Name = "double_integrator"
    DEFAULT_MATLAB_FILE = "data/double_integrator.mat"
    NUM_STATES = 2

    def __init__(self, args):
        super().__init__(args)
        
        # Initialize input bounds specific to Double Integrator
        self.input_bounds = {
            'min': torch.tensor([-args.input_max]),
            'max': torch.tensor([args.input_max])
        }
        
        # Define the boundary condition function with fixed radius
        self.boundary_radius = sqrt(0.25)
        self.boundary_fn = lambda states: double_integrator_boundary(states, self.boundary_radius)
    
    def compute_hamiltonian(self, x, p, func_map: dict) -> torch.Tensor:
        """
        Instance method that delegates to the standalone function for better serializability.
        """
        # Check if control bounds are symmetric
        input_min = self.input_bounds['min'].to(self.device)
        input_max = self.input_bounds['max'].to(self.device)

        return double_integrator_hamiltonian(
            x, p, func_map, 
            input_bounds={
                'min': input_min,
                'max': input_max
            }, 
            reach_aim=self.reach_aim
        )

    def _get_state_names(self) -> List[str]:
        """Override state names for Double Integrator."""
        return ["Position (x)", "Velocity (v)"]
