import torch
import logging
from math import sqrt
from typing import List

from care.common.base_system import DynamicalSystem
from examples.factories import register_example

logger = logging.getLogger(__name__)

def triple_integrator_boundary(states, radius=sqrt(0.25)):
    """Compute boundary values for both PyTorch tensors and dReal variables."""
    # Check if using PyTorch tensors
    using_torch = isinstance(states, torch.Tensor)
    
    if using_torch:   
        return torch.norm(states, dim=1, keepdim=True)**2 - radius**2
    else:
        x1, x2, x3 = states[0], states[1], states[2]
        return (x1 * x1 + x2 * x2 + x3 * x3) - radius**2

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