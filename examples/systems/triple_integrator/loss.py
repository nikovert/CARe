import torch
from certreach.learning.loss_functions import HJILossFunction

class TripleIntegratorLoss(HJILossFunction):
    def __init__(self, dataset, minWith='none', reachMode='backward', reachAim='reach'):
        super().__init__(dataset, minWith, reachMode, reachAim)
        self.input_max = dataset.input_max

    def compute_hamiltonian(self, x, dudx):
        p1 = dudx[..., 0]
        p2 = dudx[..., 1]
        p3 = dudx[..., 2]

        ham = p1 * x[..., 2] + p2 * x[..., 3]
        if self.reachAim == 'avoid':
            ham -= self.input_max * torch.abs(p3)
        elif self.reachAim == 'reach':
            ham += self.input_max * torch.abs(p3)
        return ham

def initialize_loss(dataset, minWith='none', reachMode='backward', reachAim='reach'):
    return TripleIntegratorLoss(dataset, minWith, reachMode, reachAim).compute_loss
