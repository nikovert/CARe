import numpy as np
import torch
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

