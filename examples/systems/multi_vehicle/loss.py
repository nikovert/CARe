import torch
from ...learning.loss_functions import HJILossFunction

class MultiVehicleCollisionLoss(HJILossFunction):
    def __init__(self, dataset, minWith='none'):
        super().__init__(dataset, minWith)
        self.velocity = dataset.velocity
        self.omega_max = dataset.omega_max
        self.numEvaders = dataset.numEvaders
        self.num_pos_states = dataset.num_pos_states
        self.alpha_angle = dataset.alpha_angle
        self.alpha_time = dataset.alpha_time

    def compute_hamiltonian(self, x, dudx):
        dudx[..., self.num_pos_states:] = dudx[..., self.num_pos_states:] / self.alpha_angle

        ham = self.velocity * (torch.cos(self.alpha_angle * x[..., self.num_pos_states + 1]) * dudx[..., 0] + 
                             torch.sin(self.alpha_angle * x[..., self.num_pos_states + 1]) * dudx[..., 1]) - \
              self.omega_max * torch.abs(dudx[..., self.num_pos_states])

        for i in range(self.numEvaders):
            theta_index = self.num_pos_states + 1 + i + 1
            xcostate_index = 2 * (i + 1)
            ycostate_index = 2 * (i + 1) + 1
            thetacostate_index = self.num_pos_states + 1 + i
            
            ham_local = self.velocity * (torch.cos(self.alpha_angle * x[..., theta_index]) * dudx[..., xcostate_index] + 
                                       torch.sin(self.alpha_angle * x[..., theta_index]) * dudx[..., ycostate_index]) + \
                       self.omega_max * torch.abs(dudx[..., thetacostate_index])
            ham = ham + ham_local

        ham = ham * self.alpha_time
        return ham

def initialize_loss(dataset, minWith='none'):
    return MultiVehicleCollisionLoss(dataset, minWith).compute_loss
