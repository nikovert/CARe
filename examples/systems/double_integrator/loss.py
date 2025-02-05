import torch
from certreach.learning.loss_functions import HJILossFunction

class DoubleIntegratorLoss(HJILossFunction):
    """Loss function specific to the Double Integrator system."""
    
    def __init__(self, dataset, minWith='none', reachMode='backward', reachAim='reach'):
        super().__init__(dataset, minWith, reachMode, reachAim)
        self.input_max = dataset.input_max

    def compute_hamiltonian(self, x, dudx):
        """Compute the Hamiltonian for the Double Integrator system."""
        p1 = dudx[..., 0]
        p2 = dudx[..., 1]

        ham = p1 * x[..., 2]
        if self.reachAim == 'avoid':
            ham -= self.input_max * torch.abs(p2)
        elif self.reachAim == 'reach':
            ham += self.input_max * torch.abs(p2)
        
        return ham

def initialize_loss(dataset, minWith='none', reachMode='backward', reachAim='reach'):
    """Factory function to create Double Integrator loss function."""
    return DoubleIntegratorLoss(dataset, minWith, reachMode, reachAim).compute_loss
