import torch
from ...learning.loss_functions import HJILossFunction

class Air3DLoss(HJILossFunction):
    def __init__(self, dataset, minWith='none'):
        super().__init__(dataset, minWith)
        self.velocity = dataset.velocity
        self.omega_max = dataset.omega_max
        self.alpha_angle = dataset.alpha_angle

    def compute_hamiltonian(self, x, dudx):
        x_theta = x[..., 3] * 1.0
        dudx[..., 2] = dudx[..., 2] / self.alpha_angle
        x_theta = self.alpha_angle * x_theta

        ham = self.omega_max * torch.abs(dudx[..., 0] * x[..., 2] - dudx[..., 1] * x[..., 1] - dudx[..., 2])
        ham = ham - self.omega_max * torch.abs(dudx[..., 2])
        ham = ham + (self.velocity * (torch.cos(x_theta) - 1.0) * dudx[..., 0]) + (self.velocity * torch.sin(x_theta) * dudx[..., 1])
        return ham

def initialize_loss(dataset, minWith="none", reachMode="forward", reachAim="reach"):
    """Initialize loss function for Air3D system"""
    return Air3DLoss(dataset, minWith, reachMode, reachAim)
