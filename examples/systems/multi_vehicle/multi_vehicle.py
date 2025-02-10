import os
import torch
import logging
import numpy as np
import torch.multiprocessing as mp
from typing import Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from certreach.common.dataset import ReachabilityDataset
from certreach.learning.training import train
from certreach.learning.networks import SingleBVPNet
from certreach.verification.symbolic import extract_symbolic_model
from certreach.verification.dreal_utils import (
    extract_dreal_partials,
    process_dreal_result
)
from certreach.verification.verify import verify_system
from certreach.common.matlab_loader import load_matlab_data, compare_with_nn

from .verification import dreal_multi_vehicle_BRS
from .loss import initialize_loss
from examples.utils.experiment_utils import get_experiment_folder, save_experiment_details
from examples.factories import register_example

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)

@register_example
class MultiVehicle:
    Name = "multi_vehicle"

    def __init__(self, args):
        self.args = args
        self.root_path = get_experiment_folder(args.logging_root, self.Name)
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(args.device)
        self.model = None
        self.dataset = None
        self.loss_fn = None
        self.num_vehicles = args.num_vehicles
        self.num_states = 3 * self.num_vehicles  # Each vehicle has [x, y, theta]

    def initialize_components(self):
        if self.dataset is None:
            def multi_vehicle_boundary(coords):
                # Compute pairwise distances between all vehicles
                boundary_values = []
                for i in range(self.num_vehicles):
                    for j in range(i + 1, self.num_vehicles):
                        # Extract positions of each vehicle pair
                        pos_i = coords[:, 1+i*3:3+i*3]  # [x, y] of vehicle i
                        pos_j = coords[:, 1+j*3:3+j*3]  # [x, y] of vehicle j
                        
                        # Compute distance between vehicles
                        dist = torch.norm(pos_i - pos_j, dim=1, keepdim=True)
                        boundary_values.append(dist - self.args.collision_radius)
                
                # Take minimum of all pairwise distances
                boundary_values = torch.cat(boundary_values, dim=1)
                return torch.min(boundary_values, dim=1, keepdim=True)[0]

            self.dataset = ReachabilityDataset(  # Changed from BaseReachabilityDataset
                numpoints=self.args.numpoints,
                tMin=self.args.tMin,
                tMax=self.args.tMax,
                pretrain=self.args.pretrain,
                pretrain_iters=self.args.pretrain_iters,
                counter_start=self.args.counter_start,
                counter_end=self.args.counter_end,
                num_src_samples=self.args.num_src_samples,
                seed=self.args.seed,
                device=self.device,
                num_states=self.num_states,
                compute_boundary_values=multi_vehicle_boundary
            )

        # Initialize model and loss function same as other systems
        if self.model is None:
            self.model = SingleBVPNet(
                in_features=1 + self.num_states,  # time + states
                out_features=1,
                type=self.args.model_type,
                mode=self.args.model_mode,
                hidden_features=self.args.num_nl,
                num_hidden_layers=self.args.num_hl,
                use_polynomial=self.args.use_polynomial,
                poly_degree=self.args.poly_degree
            ).to(self.device)

        if self.loss_fn is None:
            self.loss_fn = initialize_loss(
                self.dataset,
                minWith=self.args.minWith,
                reachMode=self.args.reachMode,
                reachAim=self.args.reachAim
            )

    def verify(self):
        """Verify the multi-vehicle system using dReal"""
        symbolic_expr = extract_symbolic_model(self.model)
        dreal_data = extract_dreal_partials(symbolic_expr, 1 + self.num_states)
        
        return verify_system(
            dreal_data=dreal_data,
            verification_fn=dreal_multi_vehicle_BRS,
            save_dir=self.root_path,
            epsilon=self.args.epsilon,
            reachMode=self.args.reachMode,
            reachAim=self.args.reachAim,
            setType=self.args.setType,
            num_vehicles=self.num_vehicles,
            velocity=self.args.velocity,
            omega_max=self.args.omega_max
        )

    def validate(self, model, ckpt_dir, epoch):
        """Validation visualization for multi-vehicle system"""
        times = [self.args.tMin, 0.5 * (self.args.tMin + self.args.tMax), self.args.tMax]
        num_times = len(times)

        # Create state space sampling grid for first two vehicles
        x_range = torch.linspace(-1.5, 1.5, 50)
        y_range = torch.linspace(-1.5, 1.5, 50)
        
        # Fixed positions for other vehicles
        fixed_states = torch.zeros(1, self.num_states - 4, device=self.device)  # -4 for first two vehicles' x,y
        fixed_states[0, 0] = 0.5  # Example fixed position for third vehicle
        fixed_states[0, 1] = 0.5

        fig, axes = plt.subplots(num_times, 1, figsize=(10, 5*num_times))
        if num_times == 1:
            axes = [axes]

        for t_idx, t in enumerate(times):
            X1, Y1 = torch.meshgrid(x_range, y_range, indexing='ij')
            
            # Create coordinates for first two vehicles
            coords = torch.cat((
                torch.ones_like(X1.reshape(-1, 1)) * t,
                X1.reshape(-1, 1),
                Y1.reshape(-1, 1),
                torch.zeros_like(X1.reshape(-1, 1)),  # theta1
                X1.reshape(-1, 1) + 0.5,  # x2 offset
                Y1.reshape(-1, 1) + 0.5,  # y2 offset
                torch.zeros_like(X1.reshape(-1, 1)),  # theta2
                fixed_states.repeat(X1.reshape(-1, 1).shape[0], 1)  # Other vehicles' states
            ), dim=1).to(self.device)

            model_out = model({'coords': coords})['model_out'].detach().cpu().numpy()
            model_out = model_out.reshape(X1.shape)

            contour = axes[t_idx].contourf(X1, Y1, model_out, levels=50, cmap='bwr')
            axes[t_idx].set_title(f"t = {t:.2f}")
            axes[t_idx].set_xlabel("Vehicle 1 x")
            axes[t_idx].set_ylabel("Vehicle 1 y")
            plt.colorbar(contour, ax=axes[t_idx])

            # Plot fixed vehicle positions
            for i in range(2, self.num_vehicles):
                x = fixed_states[0, (i-2)*3]
                y = fixed_states[0, (i-2)*3 + 1]
                axes[t_idx].plot(x, y, 'k*', markersize=10, label=f'Vehicle {i+1}')

        plt.tight_layout()
        fig.savefig(os.path.join(ckpt_dir, f'MultiVehicle_val_epoch_{epoch:04d}.png'))
        plt.close(fig)

    def plot_final_model(self, model, save_dir, epsilon, save_file="Final_Model_Comparison_With_Zero_Set.png"):
        """Plot comparison of multi-vehicle value functions"""
        x_range = torch.linspace(-1, 1, 50)
        y_range = torch.linspace(-1, 1, 50)
        
        # Fixed states for vehicles beyond the first two
        fixed_states = torch.zeros(1, self.num_states - 4, device=self.device)
        fixed_states[0, 0] = 0.5
        fixed_states[0, 1] = 0.5

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        fig.suptitle(f"Multi-Vehicle Value Function Comparison (ε={epsilon})")

        X1, Y1 = torch.meshgrid(x_range, y_range, indexing='ij')
        coords = torch.cat((
            torch.ones_like(X1.reshape(-1, 1)) * self.args.tMax,
            X1.reshape(-1, 1),
            Y1.reshape(-1, 1),
            torch.zeros_like(X1.reshape(-1, 1)),
            X1.reshape(-1, 1) + 0.5,
            Y1.reshape(-1, 1) + 0.5,
            torch.zeros_like(X1.reshape(-1, 1)),
            fixed_states.repeat(X1.reshape(-1, 1).shape[0], 1)
        ), dim=1).to(self.device)

        model_out = model({'coords': coords})['model_out'].cpu().detach().numpy().reshape(X1.shape)
        adjusted_model_out = model_out - epsilon

        # Original value function
        contour1 = axes[0].contourf(X1, Y1, model_out, levels=50, cmap='bwr')
        zero_level1 = axes[0].contour(X1, Y1, model_out, levels=[0], colors='k', linewidths=2)
        axes[0].clabel(zero_level1, inline=True, fontsize=8)
        axes[0].set_title("Original Value Function")
        axes[0].set_xlabel("Vehicle 1 x")
        axes[0].set_ylabel("Vehicle 1 y")
        fig.colorbar(contour1, ax=axes[0])

        # Epsilon-adjusted value function
        contour2 = axes[1].contourf(X1, Y1, adjusted_model_out, levels=50, cmap='bwr')
        zero_level2 = axes[1].contour(X1, Y1, adjusted_model_out, levels=[0], colors='k', linewidths=2)
        axes[1].clabel(zero_level2, inline=True, fontsize=8)
        axes[1].set_title(f"Epsilon-Adjusted Value")
        axes[1].set_xlabel("Vehicle 1 x")
        axes[1].set_ylabel("Vehicle 1 y")
        fig.colorbar(contour2, ax=axes[1])

        # Zero-level set comparison
        axes[2].contour(X1, Y1, model_out, levels=[0], colors='b', linewidths=2, linestyles='--')
        axes[2].contour(X1, Y1, adjusted_model_out, levels=[0], colors='r', linewidths=2, linestyles='-')
        axes[2].set_title("Zero-Level Set Comparison")
        axes[2].set_xlabel("Vehicle 1 x")
        axes[2].set_ylabel("Vehicle 1 y")
        axes[2].legend(["Original", "ε-Adjusted"], loc="upper right")

        # Plot fixed vehicle positions
        for ax in axes:
            for i in range(2, self.num_vehicles):
                x = fixed_states[0, (i-2)*3]
                y = fixed_states[0, (i-2)*3 + 1]
                ax.plot(x, y, 'k*', markersize=10, label=f'Vehicle {i+1}')

        plt.tight_layout()
        save_path = os.path.join(save_dir, save_file)
        plt.savefig(save_path)
        plt.close(fig)

    def train(self, counterexample: Optional[torch.Tensor] = None):
        """Train with counterexample handling"""
        self.logger.info("Initializing training components")
        
        if counterexample is not None:
            def multi_vehicle_boundary(coords):
                boundary_values = []
                for i in range(self.num_vehicles):
                    for j in range(i + 1, self.num_vehicles):
                        pos_i = coords[:, 1+i*3:3+i*3]
                        pos_j = coords[:, 1+j*3:3+j*3]
                        dist = torch.norm(pos_i - pos_j, dim=1, keepdim=True)
                        boundary_values.append(dist - self.args.collision_radius)
                boundary_values = torch.cat(boundary_values, dim=1)
                return torch.min(boundary_values, dim=1, keepdim=True)[0]

            self.dataset = ReachabilityDataset(  # Changed from BaseReachabilityDataset
                numpoints=self.args.numpoints,
                tMin=self.args.tMin,
                tMax=self.args.tMax,
                pretrain=self.args.pretrain,
                pretrain_iters=self.args.pretrain_iters,
                counter_start=self.args.counter_start,
                counter_end=self.args.counter_end,
                num_src_samples=self.args.num_src_samples,
                seed=self.args.seed,
                device=self.device,
                counterexample=counterexample,
                percentage_in_counterexample=20,
                percentage_at_t0=20,
                epsilon_radius=self.args.epsilon_radius,
                num_states=self.num_states,
                compute_boundary_values=multi_vehicle_boundary
            )
        else:
            self.initialize_components()

        # Standard training procedure
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            pin_memory=False
        )
        
        self.logger.info("Starting model training")
        train(
            model=self.model,
            train_dataloader=dataloader,
            epochs=self.args.num_epochs,
            lr=self.args.lr,
            steps_til_summary=100,
            epochs_til_checkpoint=1000,
            model_dir=self.root_path,
            loss_fn=self.loss_fn,
            clip_grad=False,
            use_lbfgs=False,
            validation_fn=self.validate,
            start_epoch=0
        )

        self.logger.info("Saving experiment details")
        save_experiment_details(self.root_path, str(self.loss_fn), vars(self.args))

    def load_model(self, model_path):
        if self.model is None:
            self.initialize_components()
        
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
