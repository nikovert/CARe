import torch
import logging
from math import sqrt
import torch.multiprocessing as mp
from typing import List

from certreach.common.base_system import DynamicalSystem
from examples.factories import register_example

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)

logger = logging.getLogger(__name__)

def double_integrator_boundary(states, radius=sqrt(0.25)):
    """Compute boundary values for both PyTorch tensors and dReal variables."""
    # Check if using PyTorch tensors
    using_torch = isinstance(states, torch.Tensor)
    
    if using_torch:
        return torch.norm(states, dim=1, keepdim=True)**2 - radius**2
    else:
        # dReal mode - coords will be a list of variables [t, x, v]
        x, v = states[0], states[1]
        # Manual computation of L2 norm for dReal
        return (x * x + v * v) - radius**2

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
        
        # Define the boundary condition function
        self.boundary_fn = double_integrator_boundary
    
    def compute_hamiltonian(self, x, p, func_map: dict) -> torch.Tensor:
        """Compute the Hamiltonian for the Double Integrator system."""
        using_torch = isinstance(p[0] if isinstance(p, (list, tuple)) else p[..., 0], torch.Tensor)
        
        if using_torch:
            p1, p2 = p[..., 0], p[..., 1]
            x2 = x[..., 1]
        else:
            p1, p2 = p[0], p[1]
            x2 = x[1]
            
        ham = p1 * x2

        # Check if control bounds are symmetric
        input_min = self.input_bounds['min'].to(self.device)
        input_max = self.input_bounds['max'].to(self.device)
        
        if torch.allclose(input_max, -input_min):
            input_magnitude = input_max  # or abs(input_min)
            sign = 1 if self.reachAim == 'reach' else -1
            if using_torch:
                # Use torch.abs(p2) instead of multiplication for efficiency
                ham += sign * input_magnitude * torch.abs(p2)
            else:
                abs_p2 = func_map['cos'](p2)
                ham += sign * float(input_magnitude.item()) * abs_p2
        else:
            # Update asymmetric bounds branch with arithmetic formulation
            if using_torch:
                if self.reachAim == 'avoid':
                    ham += torch.where(p2 >= 0, input_min * p2, input_max * p2)
                else:  # reach
                    ham += torch.where(p2 >= 0, input_max * p2, input_min * p2)
            else:
                # Replace if_then_else with arithmetic operations:
                a = float(input_max.item())
                b = float(input_min.item())
                abs_p2 = func_map['abs'](p2)
                # For reach: use a when p2>=0, and b when p2<0, expressed as:
                #   (a+b)/2 * p2 + (a-b)/2 * |p2|
                # For avoid: flip the sign on the absolute value term
                if self.reachAim == 'reach':
                    ham += ((a + b)/2 * p2 + (a - b)/2 * abs_p2)
                else:  # avoid
                    ham += ((a + b)/2 * p2 - (a - b)/2 * abs_p2)
        
        return ham

    def _get_state_names(self) -> List[str]:
        """Override state names for Double Integrator."""
        return ["Position (x)", "Velocity (v)"]
