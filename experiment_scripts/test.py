# ================================ #
# Train Double Integrator System   #
# ================================ #

# Fix Python Import Path for Cross-Module Access
import sys
import os

# Add project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Standard Imports
import torch
import modules  # Model definition
import certreach.training as training  # Training loop
import loss_functions
from dataio import DoubleIntegratorDataset

# Plotting and Argument Parsing Libraries
import matplotlib
matplotlib.use('Agg')  # Disable interactive mode for plotting
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import configargparse

# For model structure and saving models
from modules_beyond import get_experiment_folder, save_experiment_details

# ================================ #
# Argument Parser Setup            #
# ================================ #

# Create an argument parser for command-line options
p = configargparse.ArgumentParser()

# Logging and Experiment Settings
p.add_argument('--logging_root', type=str, default='./logs', help='Root directory for logging.')
p.add_argument('--experiment_name', type=str, default="DefaultExperiment", help='Name of the experiment.')

# Training Settings
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-5, help='Learning rate.')
p.add_argument('--num_epochs', type=int, default=100000, help='Number of training epochs.')
p.add_argument('--epochs_til_ckpt', type=int, default=1000, help='Checkpoint saving frequency.')
p.add_argument('--steps_til_summary', type=int, default=100, help='Logging summary frequency.')

# Model Settings
p.add_argument('--model', type=str, default='sine', choices=['sine', 'tanh', 'sigmoid', 'relu'], help='Activation type.')
p.add_argument('--mode', type=str, default='mlp', choices=['mlp', 'rbf', 'pinn'], help='Model architecture.')
p.add_argument('--in_features', type=int, default=3, help='Number of input features.')  # Added
p.add_argument('--out_features', type=int, default=1, help='Number of output features.')  # Added
p.add_argument('--tMin', type=float, default=0.0, help='Start time of the simulation.')
p.add_argument('--tMax', type=float, default=1.0, help='End time of the simulation.')
p.add_argument('--num_hl', type=int, default=0, help='Number of hidden layers.')
p.add_argument('--num_nl', type=int, default=56, help='Number of neurons per layer.')
p.add_argument('--minWith', type=str, default='none', choices=['none', 'zero', 'target'], help='Constraint type.')

# Polynomial Layer Settings
p.add_argument('--use_polynomial', action='store_true', default=False, help='Enable polynomial layer.')
p.add_argument('--poly_degree', type=int, default=2, help='Polynomial layer degree.')

# Double Integrator Specific Settings
p.add_argument('--input_max', type=float, default=1.0, help='Maximum control input (acceleration).')
p.add_argument('--pretrain', action='store_true', default=True, help='Enable pretraining mode.')
p.add_argument('--pretrain_iters', type=int, default=2000, help='Number of pretraining iterations.')
p.add_argument('--counter_start', type=int, default=0, help='Start of curriculum learning.')
p.add_argument('--counter_end', type=int, default=100e3, help='End of curriculum learning.')
p.add_argument('--num_src_samples', type=int, default=1000, help='Number of samples from initial state.')
p.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')

# Load Parsed Arguments
opt = p.parse_args()

# ================================ #
# Dataset Creation                 #
# ================================ #

# Create a dataset for the Double Integrator system
dataset = DoubleIntegratorDataset(
    numpoints=65000, tMin=opt.tMin, tMax=opt.tMax, 
    input_max=opt.input_max, pretrain=opt.pretrain,
    pretrain_iters=opt.pretrain_iters,
    counter_start=opt.counter_start, counter_end=opt.counter_end, 
    num_src_samples=opt.num_src_samples, seed=opt.seed
)

# DataLoader Initialization
dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

# ================================ #
# Model Definition                 #
# ================================ #

# Initialize the neural network model
opt.use_polynomial = True
model = modules.SingleBVPNet(
    in_features=opt.in_features, # [time, position, velocity]
    out_features=opt.out_features,
    type=opt.model, 
    mode=opt.mode,
    hidden_features=opt.num_nl,
    num_hidden_layers=opt.num_hl, 
    use_polynomial=opt.use_polynomial, 
    poly_degree=opt.poly_degree,
).cuda()

# Initialize Loss Function from loss_functions.py
loss_fn = loss_functions.initialize_hji_double_integrator(dataset, opt.minWith)

# ================================ #
# Experiment Folder Setup          #
# ================================ #

# Find or create the experiment folder
# Setup experiment folder
experiment_folder = get_experiment_folder(opt.logging_root, opt.experiment_name)
root_path = experiment_folder


# ================================ #
# Validation Function Definition   #
# ================================ #
def val_fn(model, ckpt_dir, epoch, tMin=0.0, tMax=1.0, input_max=1.0, radius=0.25):
    """
    Validation function for the Double Integrator system.

    This function evaluates the trained neural network at specified time points,
    unnormalizes the output, and visualizes the value function using contour plots.

    Args:
        model (nn.Module): The trained neural network model.
        ckpt_dir (str): Directory where validation plots are saved.
        epoch (int): The current training epoch number.
        tMin (float): Minimum time value for evaluation.
        tMax (float): Maximum time value for evaluation.
        input_max (float): Maximum control input for acceleration.
        radius (float): Radius of the zero-level set (default=0.25).
    """



    # Unnormalization constants (matching the dataset normalization)
    norm_to = 0.02  # Target normalization range
    mean = 0.25     # Mean applied during normalization
    var = 0.5       # Variance scaling factor

    # Define evaluation time points
    times = [tMin, 0.5 * (tMin + tMax), tMax]  # Start, midpoint, and end times
    num_times = len(times)

    # Define state space sampling range for position and velocity
    state_range = torch.linspace(-1.5, 1.5, 200)  # Sampling range from -1.5 to 1.5

    # Create a figure for contour plots
    fig, axes = plt.subplots(num_times, 1, figsize=(10, 15))

    for i, t in enumerate(times):
        # Create a grid of state points (position, velocity)
        X, V = torch.meshgrid(state_range, state_range, indexing='ij')

        # Flatten the state points for model evaluation
        positions = X.reshape(-1, 1)   # Flatten positions (x)
        velocities = V.reshape(-1, 1)  # Flatten velocities (v)

        # Set time points matching the current evaluation time
        time_coords = torch.ones_like(positions) * t

        # Combine time, position, and velocity into a single input tensor
        coords = torch.cat((time_coords, positions, velocities), dim=1).cuda()

        # Evaluate the trained model
        model_in = {'coords': coords}
        model_out = model(model_in)['model_out'].detach().cpu().numpy()

        # Unnormalize the model output to interpret results
        # model_out = (model_out * var / norm_to) + mean

        # Reshape the model output back into a grid for visualization
        model_out = model_out.reshape(X.shape)

        # Create a filled contour plot for the value function
        contour = axes[i].contourf(X, V, model_out, levels=50, cmap='bwr')

        # Set plot titles and labels for clarity
        axes[i].set_title(f"t = {t:.2f}")
        axes[i].set_xlabel("Position (x)")
        axes[i].set_ylabel("Velocity (v)")

        # Add a colorbar for interpretation
        fig.colorbar(contour, ax=axes[i])

    # Adjust layout and save the figure
    plt.tight_layout()
    fig.savefig(os.path.join(ckpt_dir, f'DoubleIntegrator_val_epoch_{epoch:04d}.png'))
    plt.close(fig)

# ================================ #
# Training Procedure               #
# ================================ #

# Perform training
try:
    training.train(
        model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
        steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
        model_dir=root_path, loss_fn=loss_fn, clip_grad=False,
        use_lbfgs=False, validation_fn=lambda m, d, e: val_fn(m, d, e, opt.tMin, opt.tMax, opt.input_max),
        start_epoch=0
    )
    # Save the model details
    save_experiment_details(root_path, True, str(loss_fn), vars(opt))
except KeyboardInterrupt:
    save_experiment_details(root_path, False, str(loss_fn), vars(opt))

