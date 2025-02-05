import math
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import torch
import json
import logging

logger = logging.getLogger(__name__)

def check_existing_experiment(logging_root, current_params):
    """
    Check if an experiment with matching parameters already exists.
    If parameters mismatch, print which parameters differ.

    Args:
        logging_root (str): Path to the logging root directory.
        current_params (dict): The current experiment parameters.

    Returns:
        Tuple[bool, str]: (Found, Path) if matching experiment found, else (False, None).
    """
    logger = logging.getLogger(__name__)
    experiment_folders = sorted([
        f for f in os.listdir(logging_root)
        if os.path.isdir(os.path.join(logging_root, f))
    ])

    for folder in experiment_folders:
        experiment_path = os.path.join(logging_root, folder)
        json_file_path = os.path.join(experiment_path, "experiment_details.json")

        if os.path.isfile(json_file_path):
            try:
                with open(json_file_path, 'r') as f:
                    saved_details = json.load(f)

                    # Ignore logging_root in comparison
                    saved_params = saved_details.get("model_parameters", {})
                    saved_params.pop("logging_root", None)

                    # Compare parameters excluding 'logging_root'
                    filtered_current_params = {k: v for k, v in current_params.items() if k != "logging_root"}
                    
                    if saved_params == filtered_current_params:
                        if saved_details.get("training_finished", False):
                            logger.info(f"Found matching experiment: {folder}")
                            return True, experiment_path
                        else:
                            logger.info(f"Found incomplete experiment: {folder}")
                            return False, experiment_path
                    else:
                        # Print mismatches if any
                        mismatches = {
                            k: (saved_params.get(k, "MISSING"), filtered_current_params[k])
                            for k in filtered_current_params
                            if saved_params.get(k) != filtered_current_params[k]
                        }
                        if mismatches:
                            logger.debug(f"Mismatches Found in '{folder}': {mismatches}")

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in {json_file_path}. Skipping.")
                continue

    logger.info("No matching experiment found.")
    return False, None

def get_experiment_folder(logging_root, experiment_name):
    """
    Finds or creates the next available experiment folder.

    If the latest experiment folder is incomplete (based on `experiment_details.json`),
    it will be reused. If the JSON file is missing, the folder is treated as incomplete 
    and reused as well.

    Args:
        logging_root (str): Root directory for storing experiments.
        experiment_name (str): Base name of the experiment.

    Returns:
        str: Path to the experiment folder.
    """
    logger = logging.getLogger(__name__)
    # Ensure the logging root exists
    os.makedirs(logging_root, exist_ok=True)

    # Find existing folders that match the experiment name
    folders = [f for f in os.listdir(logging_root) if f.startswith(experiment_name)]
    
    # Safe sorting function that extracts the numeric suffix if it exists
    def get_folder_number(folder_name):
        parts = folder_name.split('_')
        if len(parts) > 1:
            try:
                return int(parts[-1])
            except ValueError:
                return -1
        return -1

    folders = sorted(folders, key=get_folder_number)

    if folders:
        last_folder = folders[-1]
        last_folder_path = os.path.join(logging_root, last_folder)
        json_file_path = os.path.join(last_folder_path, "experiment_details.json")

        # Check if JSON file exists
        if os.path.isfile(json_file_path):
            with open(json_file_path, 'r') as f:
                exp_status = json.load(f)
            
            # Check if training is incomplete
            if not exp_status.get("training_finished", False):
                logger.info(f"Resuming experiment {last_folder}")
                return last_folder_path
        
        else:
            # If JSON file is missing, assume training is incomplete
            logger.warning(f"JSON file missing in {last_folder}, resuming the experiment.")
            return last_folder_path

    # Create a new experiment folder if no unfinished folder is found
    next_id = len(folders) + 1
    new_folder = f"{experiment_name}_{next_id}"
    new_folder_path = os.path.join(logging_root, new_folder)
    os.makedirs(new_folder_path, exist_ok=True)

    logger.info(f"Created new experiment folder: {new_folder}")
    return new_folder_path

def save_experiment_details(root_path, loss_fn, opt):
    """
    Save experiment details in a JSON file after training ends.

    The training is considered finished only if 'model_final.pth' exists
    in the checkpoints folder.

    Args:
        root_path (str): Path to the experiment folder.
        loss_fn (str): Name of the loss function used.
        opt (dict): Parsed arguments and configurations.
    """
    logger = logging.getLogger(__name__)
    # Check if model_final.pth exists
    final_model_path = os.path.join(root_path, "checkpoints", "model_final.pth")
    training_finished = os.path.isfile(final_model_path)

    # Create the experiment details dictionary
    experiment_details = {
        "training_finished": training_finished,
        "loss_function": loss_fn,
        "model_parameters": opt,
    }

    # Save to a JSON file
    json_file_path = os.path.join(root_path, "experiment_details.json")
    with open(json_file_path, 'w') as f:
        json.dump(experiment_details, f, indent=4)
    
    logger.info(f"Saved experiment details to {json_file_path}.")
 

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    logger.debug(f"Generating grid with sidelen={sidelen}, dim={dim}")
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def to_uint8(x):
    return (255. * x).astype(np.uint8)


def to_numpy(x):
    return x.detach().cpu().numpy()


def gaussian(x, mu=[0, 0], sigma=1e-4, d=2):
    x = x.numpy()
    if isinstance(mu, torch.Tensor):
        mu = mu.numpy()

    q = -0.5 * ((x - mu) ** 2).sum(1)
    return torch.from_numpy(1 / np.sqrt(sigma ** d * (2 * np.pi) ** d) * np.exp(q / sigma)).float()


class BaseReachabilityDataset(Dataset):
    """
    Base class for reachability analysis datasets.
    Implements common functionality for curriculum learning and data generation.
    """
    def __init__(self, numpoints, tMin=0.0, tMax=1.0, 
                 pretrain=False, pretrain_iters=2000,
                 counter_start=0, counter_end=100e3,
                 num_src_samples=1000, seed=0):
        """
        Initialize base dataset parameters.

        Args:
            numpoints (int): Number of points to sample per batch
            tMin (float): Minimum time value
            tMax (float): Maximum time value
            pretrain (bool): Whether to use pretraining
            pretrain_iters (int): Number of pretraining iterations
            counter_start (int): Start value for curriculum counter
            counter_end (int): End value for curriculum counter
            num_src_samples (int): Number of source (t=0) samples
            seed (int): Random seed for reproducibility
        """
        super().__init__()
        torch.manual_seed(seed)

        logger.debug(f"Initializing dataset with {numpoints} points")

        self.numpoints = numpoints
        self.tMin = tMin
        self.tMax = tMax
        
        self.pretrain = pretrain
        self.pretrain_iters = pretrain_iters
        self.pretrain_counter = 0

        self.counter = counter_start
        self.full_count = counter_end
        self.N_src_samples = num_src_samples
        self.num_states = None  # Child classes must set this

    def __len__(self):
        """Dataset length is always 1 as we generate data dynamically."""
        return 1

    def _get_time_samples(self):
        """Generate time samples based on training phase."""
        start_time = 0.0

        if self.pretrain:
            # During pretraining, all samples are at t=0
            time = torch.ones(self.numpoints, 1) * start_time
        else:
            # Progressive sampling during training
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(
                0, (self.tMax - self.tMin) * (self.counter / self.full_count))
            # Ensure some samples at t=0
            time[-self.N_src_samples:, 0] = start_time

        return time, start_time

    def _update_counters(self):
        """Update curriculum learning counters."""
        if self.pretrain and self.pretrain_counter == self.pretrain_iters - 1:
            logger.info("Pretraining phase completed")
        if self.pretrain:
            self.pretrain_counter += 1
            if self.pretrain_counter == self.pretrain_iters:
                self.pretrain = False
        elif self.counter < self.full_count:
            self.counter += 1

    def compute_boundary_values(self, coords):
        """
        Compute boundary values for the system.
        Must be implemented by child classes.
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        """Base implementation for getting samples."""
        # Get time samples
        time, start_time = self._get_time_samples()
        
        # Get state space samples (to be implemented by child classes)
        coords = self._sample_state_space()
        coords = torch.cat((time, coords), dim=1)

        # Compute boundary values
        boundary_values = self.compute_boundary_values(coords)

        # Create Dirichlet mask
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            dirichlet_mask = (coords[:, 0, None] == start_time)

        # Update curriculum learning counters
        self._update_counters()

        return {'coords': coords}, {
            'source_boundary_values': boundary_values,
            'dirichlet_mask': dirichlet_mask
        }

    def _sample_state_space(self):
        """
        Default implementation for sampling points in state space.
        Uniformly samples points in [-1, 1] for each state dimension.
        
        Returns:
            torch.Tensor: Sampled points of shape (numpoints, num_states)
        
        Raises:
            ValueError: If num_states is not set by child class
        """
        if self.num_states is None:
            raise ValueError("Child class must set self.num_states in __init__")
        return torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)


class ReachabilityMultiVehicleCollisionSourceNE(BaseReachabilityDataset):
    def __init__(self, numpoints, collisionR=0.25, velocity=0.6, omega_max=1.1,
                 pretrain=False, tMin=0.0, tMax=0.5, counter_start=0, counter_end=100e3,
                 numEvaders=1, pretrain_iters=2000, angle_alpha=1.0, time_alpha=1.0,
                 num_src_samples=1000):
        super().__init__(numpoints, tMin, tMax, pretrain, pretrain_iters,
                        counter_start, counter_end, num_src_samples)
        
        self.velocity = velocity
        self.omega_max = omega_max
        self.collisionR = collisionR
        self.alpha_angle = angle_alpha * math.pi
        self.alpha_time = time_alpha
        self.numEvaders = numEvaders
        self.num_states_per_vehicle = 3
        self.num_states = self.num_states_per_vehicle * (numEvaders + 1)
        self.num_pos_states = 2 * (numEvaders + 1)

    def _sample_state_space(self):
        return torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

    def compute_boundary_values(self, coords):
        # Collision cost between the pursuer and the evaders
        boundary_values = torch.norm(coords[:, 1:3] - coords[:, 3:5], dim=1, keepdim=True) - self.collisionR
        for i in range(1, self.numEvaders):
            boundary_values_current = torch.norm(coords[:, 1:3] - coords[:, 2*(i+1)+1:2*(i+1)+3], dim=1, keepdim=True) - self.collisionR
            boundary_values = torch.min(boundary_values, boundary_values_current)
        
        # Collision cost between the evaders themselves
        for i in range(self.numEvaders):
            for j in range(i+1, self.numEvaders):
                evader1_coords_index = 1 + (i+1)*2
                evader2_coords_index = 1 + (j+1)*2
                boundary_values_current = torch.norm(coords[:, evader1_coords_index:evader1_coords_index+2] - coords[:, evader2_coords_index:evader2_coords_index+2], dim=1, keepdim=True) - self.collisionR
                boundary_values = torch.min(boundary_values, boundary_values_current)

        # Normalize
        norm_to, mean, var = 0.02, 0.25, 0.5
        return (boundary_values - mean)*norm_to/var


class ReachabilityAir3DSource(BaseReachabilityDataset):
    def __init__(self, numpoints, collisionR=0.25, velocity=0.6, omega_max=1.1,
                 pretrain=False, tMin=0.0, tMax=0.5, counter_start=0, counter_end=100e3,
                 pretrain_iters=2000, angle_alpha=1.0, num_src_samples=1000, seed=0):
        super().__init__(numpoints, tMin, tMax, pretrain, pretrain_iters,
                        counter_start, counter_end, num_src_samples, seed)
        
        self.velocity = velocity
        self.omega_max = omega_max
        self.collisionR = collisionR
        self.alpha_angle = angle_alpha * math.pi
        self.num_states = 3

    def _sample_state_space(self):
        return torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

    def compute_boundary_values(self, coords):
        boundary_values = torch.norm(coords[:, 1:3], dim=1, keepdim=True) - self.collisionR
        norm_to, mean, var = 0.02, 0.25, 0.5
        return (boundary_values - mean)*norm_to/var


class DoubleIntegratorDataset(BaseReachabilityDataset):
    """
    A dataset class for the Double Integrator system.
    System Dynamics:
        State: [position (x), velocity (v)]
        Control: acceleration (u)
        Dynamics: 
            dx/dt = v
            dv/dt = u
    """

    def __init__(self, numpoints, tMin=0.0, tMax=1.0, 
                 input_max=1.0, pretrain=False, pretrain_iters=2000, 
                 counter_start=0, counter_end=100e3, 
                 num_src_samples=1000, seed=0):
        """
        Initialize the Double Integrator dataset.

        Args:
            numpoints (int): Number of points to sample per batch.
            tMin (float): Minimum time value.
            tMax (float): Maximum time value.
            input_max (float): Maximum control input (acceleration).
            pretrain (bool): Enable pretraining.
            pretrain_iters (int): Number of pretraining iterations.
            counter_start (int): Initial time counter for curriculum training.
            counter_end (int): Final time counter for curriculum training.
            num_src_samples (int): Number of initial source samples.
            seed (int): Random seed for reproducibility.
        """
        super().__init__(numpoints, tMin, tMax, pretrain, pretrain_iters,
                        counter_start, counter_end, num_src_samples, seed)

        self.input_max = input_max  # Maximum acceleration input
        self.num_states = 2  # Number of states: [position, velocity]

    def compute_boundary_values(self, coords):
        """
        Compute the initial value function for the Double Integrator system,
        using the same logic as the Air3D system.

        This defines a zero-level set as a circle of radius `radius`
        centered at the origin (x, v) = (0, 0).

        Args:
            coords (torch.Tensor): Sampled state coordinates (t, x, v).
            radius (float): Radius of the zero-level set (default=1.0).

        Returns:
            torch.Tensor: Computed boundary values using Air3D logic.
        """
        # Extract state variables [x, v] from coordinates
        pos = coords[:, 1:3]  # Extract [x, v]

        # Compute Euclidean distance from the origin (x=0, v=0)
        boundary_values = torch.norm(pos, dim=1, keepdim=True)

        # Define zero-level set like Air3D: distance from the origin minus radius
        boundary_values -= 0.25

        return boundary_values


class DubinsCarDataset(BaseReachabilityDataset):
    """
    Dataset for Dubins Car Reachability Analysis.

    System Dynamics:
        - State: [x, y, theta]
        - Control: [v, omega]
    Dynamics Equations:
        dx/dt = v * cos(theta)
        dy/dt = v * sin(theta)
        dtheta/dt = omega
    """

    def __init__(self, numpoints, tMin=0.0, tMax=1.0, 
                 velocity=0.6, omega_max=1.1, pretrain=False, 
                 pretrain_iters=2000, counter_start=0, counter_end=100e3, 
                 num_src_samples=1000, seed=0, collision_radius=0.25):
        """
        Initialize the dataset.

        Args:
            numpoints (int): Number of points to sample per batch.
            tMin (float): Minimum time value for sampling.
            tMax (float): Maximum time value for sampling.
            velocity (float): Constant forward velocity of the car.
            omega_max (float): Maximum angular velocity.
            pretrain (bool): Whether to enable pretraining.
            pretrain_iters (int): Number of pretraining iterations.
            counter_start (int): Starting time counter for curriculum learning.
            counter_end (int): End time counter for curriculum learning.
            num_src_samples (int): Number of initial samples at t=0.
            seed (int): Random seed for reproducibility.
            collision_radius (float): Radius of the obstacle for boundary conditions.
        """
        super().__init__(numpoints, tMin, tMax, pretrain, pretrain_iters,
                        counter_start, counter_end, num_src_samples, seed)

        self.velocity = velocity
        self.omega_max = omega_max
        self.collision_radius = collision_radius
        self.num_states = 3  # Number of states: [x, y, theta]

    def compute_boundary_values(self, coords):
        """
        Compute the cylindrical boundary values for the Dubins car.

        Args:
            coords (torch.Tensor): Sampled state coordinates (t, x, y, theta).

        Returns:
            torch.Tensor: Boundary values representing a cylinder in (x, y).
        """
        # Compute the Euclidean distance in (x, y) space
        distance_xy = torch.norm(coords[:, 1:3], dim=1, keepdim=True)

        # Subtract the collision radius to create a cylindrical boundary
        return distance_xy - self.collision_radius


class ThreeStateSystemDataset(BaseReachabilityDataset):
    """
    A dataset class for a 3-state system with 1 control input.
    System Dynamics:
        State: [position (x1), velocity (x2), auxiliary (x3)]
        Control: [u]
        Dynamics:
            dx1/dt = x2
            dx2/dt = -k1 * x1 - c1 * x2 + u
            dx3/dt = -k2 * x3 + c2 * x1
    """

    def __init__(self, numpoints, tMin=0.0, tMax=1.0, 
                 input_max=1.0, pretrain=False, pretrain_iters=2000, 
                 counter_start=0, counter_end=100e3, 
                 num_src_samples=1000, seed=0, radius=0.25):
        """
        Initialize the 3-state system dataset.

        Args:
            numpoints (int): Number of points to sample per batch.
            tMin (float): Minimum time value.
            tMax (float): Maximum time value.
            input_max (float): Maximum control input (u).
            pretrain (bool): Enable pretraining.
            pretrain_iters (int): Number of pretraining iterations.
            counter_start (int): Initial time counter for curriculum training.
            counter_end (int): Final time counter for curriculum training.
            num_src_samples (int): Number of initial source samples.
            seed (int): Random seed for reproducibility.
            radius (float): Radius of the zero-level set (default=0.25).
        """
        super().__init__(numpoints, tMin, tMax, pretrain, pretrain_iters,
                        counter_start, counter_end, num_src_samples, seed)

        self.input_max = input_max  # Maximum control input
        self.radius = radius  # Radius for boundary condition
        self.num_states = 3  # Number of states: [x1, x2, x3]

    def compute_boundary_values(self, coords):
        """
        Compute the initial value function for the 3-state system.

        Args:
            coords (torch.Tensor): Sampled state coordinates (t, x1, x2, x3).

        Returns:
            torch.Tensor: Computed boundary values.
        """
        # Extract state variables [x1, x2, x3] from coordinates
        pos = coords[:, 1:4]  # Extract [x1, x2, x3]

        # Compute Euclidean distance from the origin (x1=0, x2=0, x3=0)
        boundary_values = torch.norm(pos, dim=1, keepdim=True)

        # Define zero-level set: distance from the origin minus radius
        boundary_values -= self.radius

        return boundary_values


class TripleIntegratorDataset(BaseReachabilityDataset):
    """
    A dataset class for the Triple Integrator system.
    System Dynamics:
        State: [position (x1), velocity (x2), acceleration (x3)]
        Control: jerk (u)
        Dynamics: 
            dx1/dt = x2
            dx2/dt = x3
            dx3/dt = u
    """

    def __init__(self, numpoints, tMin=0.0, tMax=1.0, 
                 input_max=1.0, pretrain=False, pretrain_iters=2000, 
                 counter_start=0, counter_end=100e3, 
                 num_src_samples=1000, seed=0, radius=0.25):
        """
        Initialize the Triple Integrator dataset.

        Args:
            numpoints (int): Number of points to sample per batch.
            tMin (float): Minimum time value.
            tMax (float): Maximum time value.
            input_max (float): Maximum control input (jerk).
            pretrain (bool): Enable pretraining.
            pretrain_iters (int): Number of pretraining iterations.
            counter_start (int): Initial time counter for curriculum training.
            counter_end (int): Final time counter for curriculum training.
            num_src_samples (int): Number of initial source samples.
            seed (int): Random seed for reproducibility.
            radius (float): Radius for boundary value computation.
        """
        super().__init__(numpoints, tMin, tMax, pretrain, pretrain_iters,
                        counter_start, counter_end, num_src_samples, seed)

        self.input_max = input_max  # Maximum jerk input
        self.radius = radius
        self.num_states = 3  # Number of states: [x1 (position), x2 (velocity), x3 (acceleration)]

    def compute_boundary_values(self, coords):
        """
        Compute the initial value function for the Triple Integrator system.

        This defines a zero-level set as a sphere of radius `radius`
        centered at the origin (x1, x2, x3) = (0, 0, 0).

        Args:
            coords (torch.Tensor): Sampled state coordinates (t, x1, x2, x3).

        Returns:
            torch.Tensor: Computed boundary values using Euclidean distance.
        """
        # Extract state variables [x1, x2, x3] from coordinates
        pos = coords[:, 1:]  # Extract [x1, x2, x3]

        # Compute Euclidean distance from the origin (x1=0, x2=0, x3=0)
        boundary_values = torch.norm(pos, dim=1, keepdim=True)

        # Define zero-level set: distance from the origin minus radius
        boundary_values -= self.radius

        return boundary_values

