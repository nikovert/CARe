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
        using_torch = isinstance(p[0] if isinstance(p, (list, tuple)) else p[..., 0], torch.Tensor)
        
        if using_torch:
            p1, p2 = p[..., 0], p[..., 1]
            x1, x2 = x[..., 0], x[..., 1]
        else:
            p1, p2 = p[0], p[1]
            x1, x2 = x[0], x[1]
            
        ham = p1 * x2

        # Check if control bounds are symmetric
        if torch.allclose(self.input_max, -self.input_min):
            input_magnitude = self.input_max  # or abs(self.input_min)
            sign = 1 if self.reachAim == 'reach' else -1
            if using_torch:
                # Use torch.abs(p2) instead of multiplication for efficiency
                ham += sign * input_magnitude * torch.abs(p2)
            else:
                # Use dreal.Max to compute absolute value: |p2| = Max(p2, -p2)
                abs_p2 = dreal.Max(p2, -p2)
                ham += sign * float(input_magnitude.item()) * abs_p2
        else:
            # Update asymmetric bounds branch with arithmetic formulation
            if using_torch:
                if self.reachAim == 'avoid':
                    ham += torch.where(p2 >= 0, self.input_min * p2, self.input_max * p2)
                else:  # reach
                    ham += torch.where(p2 >= 0, self.input_max * p2, self.input_min * p2)
            else:
                # Replace if_then_else with arithmetic operations:
                a = float(self.input_max.item())
                b = float(self.input_min.item())
                abs_p2 = dreal.Max(p2, -p2)
                # For reach: use a when p2>=0, and b when p2<0, expressed as:
                #   (a+b)/2 * p2 + (a-b)/2 * |p2|
                # For avoid: flip the sign on the absolute value term
                if self.reachAim == 'reach':
                    ham += ((a + b)/2 * p2 + (a - b)/2 * abs_p2)
                else:  # avoid
                    ham += ((a + b)/2 * p2 - (a - b)/2 * abs_p2)
        
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
