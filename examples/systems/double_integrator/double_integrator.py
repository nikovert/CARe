import os
import torch
import logging
import torch.multiprocessing as mp
from typing import Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from certreach.common.dataset import ReachabilityDataset
from certreach.common.dataio import (
    get_experiment_folder,
    save_experiment_details
)
from certreach.learning.training import train
from certreach.learning.networks import SingleBVPNet
from certreach.verification.symbolic import extract_symbolic_model
from certreach.verification.dreal_utils import (
    extract_dreal_partials,
    process_dreal_result
)
from certreach.verification.verify import verify_system
from .verification import dreal_double_integrator_BRS
from .loss import initialize_loss

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)

def double_integrator_boundary(coords):
        pos = coords[:, 1:3]  # Extract [x, v]
        boundary_values = torch.norm(pos, dim=1, keepdim=True)
        return boundary_values - 0.25

class DoubleIntegrator:
    Name = "double_integrator"

    def __init__(self, args):
        self.args = args
        self.root_path = get_experiment_folder(args.logging_root, self.Name)
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(args.device)
        
        # Initialize model and other components only when needed
        self.model = None
        self.dataset = None
        self.loss_fn = None
        self.verification_fn = dreal_double_integrator_BRS  # Add this line

    def initialize_components(self, counterexample: Optional[torch.Tensor] = None):
        """Initialize dataset, model, and loss function"""
        if self.dataset is None and counterexample is None:
            self.dataset = ReachabilityDataset(
                    numpoints=85000,
                    tMin=self.args.tMin,
                    tMax=self.args.tMax,
                    pretrain=self.args.pretrain,
                    pretrain_iters=self.args.pretrain_iters,
                    counter_start=self.args.counter_start,
                    counter_end=self.args.counter_end,
                    num_src_samples=self.args.num_src_samples,
                    seed=self.args.seed,
                    device=self.device,
                    num_states=2,  # [position, velocity]
                    compute_boundary_values=double_integrator_boundary
                )
        # Initialize or update dataset with counterexample
        elif counterexample is not None:
            if not isinstance(counterexample, torch.Tensor):
                raise TypeError("counterexample must be a torch.Tensor")
            
            # Ensure counterexample has correct shape [N, 3] where 3 = [time, position, velocity]
            if counterexample.dim() == 1:
                counterexample = counterexample.unsqueeze(0)  # Add batch dimension if missing
            
            if counterexample.size(1) == 2:  # If only [position, velocity] provided
                # Add time dimension initialized to tMax (assuming worst-case scenario)
                time_dim = torch.full((counterexample.size(0), 1), self.args.tMax, device=counterexample.device)
                counterexample = torch.cat([time_dim, counterexample], dim=1)

            # Create new dataset instance with counterexample using existing numpoints
            self.dataset = ReachabilityDataset(
                numpoints=85000,  # Use existing dataset's numpoints
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
                num_states=2,  # [position, velocity]
                compute_boundary_values=double_integrator_boundary
            )
               
        if self.model is None:
            self.model = SingleBVPNet(
                in_features=self.args.in_features,
                out_features=self.args.out_features,
                type=self.args.model_type,  # Changed from model to model_type
                mode=self.args.model_mode,  # Changed from mode to model_mode
                hidden_features=self.args.num_nl,
                num_hidden_layers=self.args.num_hl,
                use_polynomial=self.args.use_polynomial,
                poly_degree=self.args.poly_degree
            ).to(self.device)

        if self.loss_fn is None:
            # Construct input bounds dictionary
            input_bounds = {
                'min': torch.tensor([-self.args.input_max]),  # Assuming symmetric bounds
                'max': torch.tensor([self.args.input_max])
            }
            
            self.loss_fn = initialize_loss(
                self.device,
                input_bounds=input_bounds,
                minWith=self.args.minWith,
                reachMode=self.args.reachMode,
                reachAim=self.args.reachAim
            )

    def train(self, counterexample: Optional[torch.Tensor] = None):
        """Train the model with optional counterexample handling."""
        self.logger.info("Initializing training")
        self.initialize_components(counterexample)
        
        # Setup data loader and continue with training
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

    def validate(self, model, ckpt_dir, epoch):
        """Validation function called during training"""
        # Unnormalization constants
        norm_to = 0.02
        mean = 0.25
        var = 0.5

        # Define evaluation time points
        times = [self.args.tMin, 0.5 * (self.args.tMin + self.args.tMax), self.args.tMax]
        num_times = len(times)

        # Create state space sampling grid
        state_range = torch.linspace(-1.5, 1.5, 200)
        fig, axes = plt.subplots(num_times, 1, figsize=(10, 15))

        for i, t in enumerate(times):
            X, V = torch.meshgrid(state_range, state_range, indexing='ij')
            positions = X.reshape(-1, 1)
            velocities = V.reshape(-1, 1)
            time_coords = torch.ones_like(positions) * t

            coords = torch.cat((time_coords, positions, velocities), dim=1).to(self.device)
            model_in = {'coords': coords}
            model_out = model(model_in)['model_out'].detach().cpu().numpy()
            model_out = model_out.reshape(X.shape)

            contour = axes[i].contourf(X, V, model_out, levels=50, cmap='bwr')
            axes[i].set_title(f"t = {t:.2f}")
            axes[i].set_xlabel("Position (x)")
            axes[i].set_ylabel("Velocity (v)")
            fig.colorbar(contour, ax=axes[i])

        plt.tight_layout()
        fig.savefig(os.path.join(ckpt_dir, f'DoubleIntegrator_val_epoch_{epoch:04d}.png'))
        plt.close(fig)

    def plot_final_model(self, model, save_dir, epsilon, save_file="Final_Model_Comparison_With_Zero_Set.png"):
        """Plot comparison of value functions"""
        state_range = torch.linspace(-1, 1, 200)
        X, V = torch.meshgrid(state_range, state_range, indexing='ij')
        positions = X.reshape(-1, 1)
        velocities = V.reshape(-1, 1)
        time_coords = torch.ones_like(positions) * self.args.tMax

        # Move coordinates to the same device as the model
        coords = torch.cat((time_coords, positions, velocities), dim=1).to(self.device)
        model_in = {'coords': coords}
        
        # Get output and move back to CPU for plotting
        model_out = model(model_in)['model_out'].cpu().detach().numpy().reshape(X.shape)
        adjusted_model_out = model_out - epsilon

        # Create comparison plots
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))

        # Original Value Function
        contour1 = axes[0].contourf(X, V, model_out, levels=50, cmap='bwr')
        zero_level1 = axes[0].contour(X, V, model_out, levels=[0], colors='k', linewidths=2)
        axes[0].clabel(zero_level1, inline=True, fontsize=8)
        axes[0].set_title("Original Value Function")
        axes[0].set_xlabel("Position (x)")
        axes[0].set_ylabel("Velocity (v)")
        fig.colorbar(contour1, ax=axes[0])

        # Epsilon-Adjusted Value Function
        contour2 = axes[1].contourf(X, V, adjusted_model_out, levels=50, cmap='bwr')
        zero_level2 = axes[1].contour(X, V, adjusted_model_out, levels=[0], colors='k', linewidths=2)
        axes[1].clabel(zero_level2, inline=True, fontsize=8)
        axes[1].set_title(f"Epsilon-Adjusted Value ($\epsilon$={epsilon})")
        axes[1].set_xlabel("Position (x)")
        axes[1].set_ylabel("Velocity (v)")
        fig.colorbar(contour2, ax=axes[1])

        # Zero-Level Set Comparison
        axes[2].contour(X, V, model_out, levels=[0], colors='b', linewidths=2, linestyles='--')
        axes[2].contour(X, V, adjusted_model_out, levels=[0], colors='r', linewidths=2, linestyles='-')
        axes[2].set_title("Zero-Level Set Comparison")
        axes[2].set_xlabel("Position (x)")
        axes[2].set_ylabel("Velocity (v)")
        axes[2].legend(["Original", f"Epsilon-Adjusted ($\epsilon$={epsilon})"], loc="upper right")

        plt.tight_layout()
        save_path = os.path.join(save_dir, save_file)
        plt.savefig(save_path)
        plt.close(fig)
        self.logger.debug(f"Saved comparison plot at: {save_path}")

    def load_model(self, model_path):
        """Load a model with proper device handling."""
        if self.model is None:
            self.initialize_components()
        
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
