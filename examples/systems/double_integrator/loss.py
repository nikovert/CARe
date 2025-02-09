import torch
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

    def compute_hamiltonian(self, x, dudx):
        """Compute the Hamiltonian for the Double Integrator system."""
        p1 = dudx[..., 0]
        p2 = dudx[..., 1]

        ham = p1 * x[..., 2]
        
        # For double integrator, we have one input affecting p2
        if self.reachAim == 'avoid':
            # For avoid, we take min over u ∈ [umin, umax]
            ham += torch.where(p2 >= 0, 
                             self.input_min * p2,
                             self.input_max * p2)
        elif self.reachAim == 'reach':
            # For reach, we take max over u ∈ [umin, umax]
            ham += torch.where(p2 >= 0, 
                             self.input_max * p2,
                             self.input_min * p2)
        
        return ham

def initialize_loss(dataset, input_bounds, minWith='none', reachMode='backward', reachAim='reach'):
    """
    Factory function to create Double Integrator loss function.
    
    Args:
        input_bounds (dict): Dictionary with 'min' and 'max' tensors for input bounds
    """
    return DoubleIntegratorLoss(dataset, input_bounds, minWith, reachMode, reachAim).compute_loss
