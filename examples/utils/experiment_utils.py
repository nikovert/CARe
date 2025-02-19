import os
import json
import logging

logger = logging.getLogger(__name__)

def get_experiment_folder(logging_root, experiment_name):
    """
    Finds the next available experiment folder name and returns the previous folder path.
    """
    # Find existing folders that match the experiment name
    if os.path.exists(logging_root):
        folders = [f for f in os.listdir(logging_root) if f.startswith(experiment_name)]
    else:
        folders = []
    
    # Sort folders by their numeric suffix
    numbered_folders = []
    for folder in folders:
        try:
            num = int(folder.split('_')[-1])
            numbered_folders.append((num, folder))
        except (ValueError, IndexError):
            continue
    
    numbered_folders.sort(reverse=True)
    
    # Previous folder is the highest numbered one (for loading models)
    prev_folder_path = None
    if numbered_folders:
        prev_folder = numbered_folders[0][1]
        prev_folder_path = os.path.join(logging_root, prev_folder)

    # Calculate next folder name
    next_id = (numbered_folders[0][0] + 1) if numbered_folders else 1
    new_folder = f"{experiment_name}_{next_id}"
    new_folder_path = os.path.join(logging_root, new_folder)
    return new_folder_path, False, prev_folder_path

def save_experiment_details(root_path, loss_fn, opt):
    """Save experiment details in a JSON file."""
    # Check if model_final.pth exists
    final_model_path = os.path.join(root_path, "checkpoints", "model_final.pth")
    training_finished = os.path.isfile(final_model_path)

    experiment_details = {
        "training_finished": training_finished,
        "loss_function": loss_fn,
        "model_parameters": opt,
    }

    json_file_path = os.path.join(root_path, "experiment_details.json")
    with open(json_file_path, 'w') as f:
        json.dump(experiment_details, f, indent=4)
    
    logger.info(f"Saved experiment details to {json_file_path}")

def setup_experiment_folder(exp_folder_path, create=False):
    """Set up experiment folder structure."""
    if create:
        os.makedirs(exp_folder_path, exist_ok=True)
    return exp_folder_path  # Return just the main folder path now
