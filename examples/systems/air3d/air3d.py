import os
import torch
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional
from certreach.common.dataset import ReachabilityDataset
from certreach.common.dataio import get_experiment_folder, save_experiment_details
from certreach.learning.training import train
from certreach.learning.networks import SingleBVPNet
from certreach.verification.symbolic import extract_symbolic_model
from certreach.verification.dreal_utils import extract_dreal_partials, process_dreal_result
from certreach.verification.verify import verify_system
from .verification import dreal_air3d_BRS
from .loss import initialize_loss

class Air3D:
    Name = "air3d"

    def __init__(self, args):
        self.args = args
        self.root_path = get_experiment_folder(args.logging_root, self.Name)
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(args.device)
        self.model = None
        self.dataset = None
        self.loss_fn = None

    def initialize_components(self):
        if self.dataset is None:
            def air3d_boundary(coords):
                pos = coords[:, 1:3]  # Extract [x, y] (ignore theta)
                boundary_values = torch.norm(pos, dim=1, keepdim=True)
                return boundary_values - self.args.collision_radius

            self.dataset = BaseReachabilityDataset(
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
                num_states=3,  # [x, y, theta]
                compute_boundary_values=air3d_boundary
            )

        if self.model is None:
            self.model = SingleBVPNet(
                in_features=self.args.in_features,
                out_features=self.args.out_features,
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

    def train(self, counterexample: Optional[torch.Tensor] = None):
        """Train the model with optional counterexample handling."""
        self.logger.info("Initializing training components")
        
        if counterexample is not None:
            def air3d_boundary(coords):
                pos = coords[:, 1:3]  # Extract [x, y] (ignore theta)
                boundary_values = torch.norm(pos, dim=1, keepdim=True)
                return boundary_values - self.args.collision_radius

            self.dataset = BaseReachabilityDataset(
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
                num_states=3,  # [x, y, theta]
                compute_boundary_values=air3d_boundary
            )
        else:
            self.initialize_components()

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

    def verify(self):
        """Verify the Air3D system using dReal"""
        symbolic_expr = extract_symbolic_model(self.model)
        dreal_data = extract_dreal_partials(symbolic_expr, self.args.in_features)
        
        return verify_system(
            dreal_data=dreal_data,
            verification_fn=dreal_air3d_BRS,
            save_dir=self.root_path,
            epsilon=self.args.epsilon,
            reachMode=self.args.reachMode,
            reachAim=self.args.reachAim,
            setType=self.args.setType,
            velocity=self.args.velocity,
            omega_max=self.args.omega_max
        )

    def validate(self, model, ckpt_dir, epoch):
        """Validation function for Air3D system"""
        times = [self.args.tMin, 0.5 * (self.args.tMin + self.args.tMax), self.args.tMax]
        num_times = len(times)

        # Create state space sampling grid
        x_range = torch.linspace(-1.5, 1.5, 100)
        y_range = torch.linspace(-1.5, 1.5, 100)
        theta_slices = [-np.pi/2, 0, np.pi/2]

        fig, axes = plt.subplots(num_times, len(theta_slices), figsize=(15, 5*num_times))

        for t_idx, t in enumerate(times):
            for theta_idx, theta in enumerate(theta_slices):
                X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
                coords = torch.cat((
                    torch.ones_like(X.reshape(-1, 1)) * t,
                    X.reshape(-1, 1),
                    Y.reshape(-1, 1),
                    torch.ones_like(X.reshape(-1, 1)) * theta
                ), dim=1).to(self.device)

                model_out = model({'coords': coords})['model_out'].detach().cpu().numpy()
                model_out = model_out.reshape(X.shape)

                # Plot results
                ax = axes[t_idx, theta_idx]
                contour = ax.contourf(X, Y, model_out, levels=50, cmap='bwr')
                ax.set_title(f"t={t:.2f}, θ={theta:.2f}")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                fig.colorbar(contour, ax=ax)

        plt.tight_layout()
        fig.savefig(os.path.join(ckpt_dir, f'Air3D_val_epoch_{epoch:04d}.png'))
        plt.close(fig)

    def plot_final_model(self, model, save_dir, epsilon, save_file="Final_Model_Comparison_With_Zero_Set.png"):
        """Plot comparison of Air3D value functions"""
        x_range = torch.linspace(-1, 1, 100)
        y_range = torch.linspace(-1, 1, 100)
        theta = 0  # Middle slice for comparison

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        fig.suptitle(f"Air3D Value Function Comparison (ε={epsilon})")

        X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
        coords = torch.cat((
            torch.ones_like(X.reshape(-1, 1)) * self.args.tMax,
            X.reshape(-1, 1),
            Y.reshape(-1, 1),
            torch.ones_like(X.reshape(-1, 1)) * theta
        ), dim=1).to(self.device)

        # Get model output and adjust
        model_out = model({'coords': coords})['model_out'].cpu().detach().numpy().reshape(X.shape)
        adjusted_model_out = model_out - epsilon

        # Original value function
        contour1 = axes[0].contourf(X, Y, model_out, levels=50, cmap='bwr')
        zero_level1 = axes[0].contour(X, Y, model_out, levels=[0], colors='k', linewidths=2)
        axes[0].clabel(zero_level1, inline=True, fontsize=8)
        axes[0].set_title("Original Value Function")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(contour1, ax=axes[0])

        # Epsilon-adjusted value function
        contour2 = axes[1].contourf(X, Y, adjusted_model_out, levels=50, cmap='bwr')
        zero_level2 = axes[1].contour(X, Y, adjusted_model_out, levels=[0], colors='k', linewidths=2)
        axes[1].clabel(zero_level2, inline=True, fontsize=8)
        axes[1].set_title(f"Epsilon-Adjusted Value")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        fig.colorbar(contour2, ax=axes[1])

        # Zero-level set comparison
        axes[2].contour(X, Y, model_out, levels=[0], colors='b', linewidths=2, linestyles='--')
        axes[2].contour(X, Y, adjusted_model_out, levels=[0], colors='r', linewidths=2, linestyles='-')
        axes[2].set_title("Zero-Level Set Comparison")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].legend(["Original", "ε-Adjusted"], loc="upper right")

        plt.tight_layout()
        save_path = os.path.join(save_dir, save_file)
        plt.savefig(save_path)
        plt.close(fig)

    def load_model(self, model_path):
        """Load a model with proper device handling."""
        if self.model is None:
            self.initialize_components()
        
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
