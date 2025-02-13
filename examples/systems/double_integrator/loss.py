import torch
import dreal
from certreach.learning.loss_functions import HJILossFunction

class DoubleIntegratorLoss(HJILossFunction):
    """Loss function specific to the Double Integrator system."""
    
    def __init__(self, device, input_bounds, minWith='none', reachMode='backward', reachAim='reach'):
        """
        Args:
            input_bounds (dict): Dictionary containing 'min' and 'max' tensors for input bounds
                               Shape of each tensor should match input dimensionality
        """
        super().__init__(minWith, reachMode, reachAim)
        if not isinstance(input_bounds, dict) or 'min' not in input_bounds or 'max' not in input_bounds:
            raise ValueError("input_bounds must be a dict with 'min' and 'max' keys")
        
        # Move input bounds to the dataset's device
        self.input_min = input_bounds['min'].to(device)
        self.input_max = input_bounds['max'].to(device)

    def compute_hamiltonian(self, x, p):
        """Compute the Hamiltonian for the Double Integrator system."""
        # Check if inputs are PyTorch tensors
        using_torch = isinstance(p[0] if isinstance(p, (list, tuple)) else p[..., 0], torch.Tensor)
        
        # Extract components based on input type
        if using_torch:
            p1, p2 = p[..., 0], p[..., 1]
            x1, x2 = x[..., 0], x[..., 1]
        else:
            # dReal mode
            p1, p2 = p[0], p[1]
            x1, x2 = x[0], x[1]
            
        ham = p1 * x2

        if using_torch:
            if self.reachAim == 'avoid':
                ham += torch.where(p2 >= 0, 
                                 self.input_min * p2,
                                 self.input_max * p2)
            elif self.reachAim == 'reach':
                ham += torch.where(p2 >= 0, 
                                 self.input_max * p2,
                                 self.input_min * p2)
        else:
            input_min = float(self.input_min.item())
            input_max = float(self.input_max.item())
            
            # Use dreal.if_then_else for conditional expressions
            if self.reachAim == 'avoid':
                return ham + dreal.if_then_else(p2 >= 0, input_min * p2, input_max * p2)
            elif self.reachAim == 'reach':
                return ham + dreal.if_then_else(p2 >= 0, input_max * p2, input_min * p2)
            
        return ham

def initialize_loss(dataset, input_bounds, minWith='none', reachMode='backward', reachAim='reach'):
    """
    Factory function to create Double Integrator loss function.
    
    Args:
        input_bounds (dict): Dictionary with 'min' and 'max' tensors for input bounds
    """
    return DoubleIntegratorLoss(dataset, input_bounds, minWith, reachMode, reachAim).compute_loss

def initialize_hamiltonian(dataset, input_bounds, minWith='none', reachMode='backward', reachAim='reach'):
    """
    Factory function to create Double Integrator loss function.
    
    Args:
        input_bounds (dict): Dictionary with 'min' and 'max' tensors for input bounds
    """
    return DoubleIntegratorLoss(dataset, input_bounds, minWith, reachMode, reachAim).compute_hamiltonian
