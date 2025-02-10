import configargparse
import json
import os
import atexit
import multiprocessing
import logging
from examples.log import configure_logging
import torch
from certreach.verification.cegis import CEGISLoop
from examples.factories import create_example, EXAMPLE_NAMES
from examples.utils.experiment_utils import get_experiment_folder, save_experiment_details, setup_experiment_folder

def parse_args():
    """Parse command line arguments."""
    p = configargparse.ArgumentParser(description='Train and verify neural networks for reachability analysis')
    
    # Required arguments
    p.add_argument('--example', type=str, required=True, choices=EXAMPLE_NAMES,
                       help='Name of the example to run')
    p.add_argument('--run_mode', type=str, required=True, choices=['train', 'verify', 'cegis'],
                       help='Mode to run: train, verify, or cegis')
    
    # Logging and Experiment Settings
    p.add_argument('--logging_root', type=str, default='./logs', help='Root directory for logging.')
    p.add_argument('--experiment_name', type=str, default="Double_integrator", help='Name of the experiment.')

    # Training Settings
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-5, help='Learning rate.')
    p.add_argument('--num_epochs', type=int, default=30000, help='Number of training epochs.')
    p.add_argument('--epochs_til_ckpt', type=int, default=1000, help='Checkpoint saving frequency.')
    p.add_argument('--steps_til_summary', type=int, default=100, help='Logging summary frequency.')

    # Model Settings
    p.add_argument('--model_type', type=str, default='sine', choices=['sine', 'tanh', 'sigmoid', 'relu'])
    p.add_argument('--model_mode', type=str, default='mlp', choices=['mlp', 'rbf', 'pinn'])
    p.add_argument('--in_features', type=int, default=3)
    p.add_argument('--out_features', type=int, default=1)
    p.add_argument('--num_hl', type=int, default=0)
    p.add_argument('--num_nl', type=int, default=32)
    p.add_argument('--use_polynomial', action='store_true', default=False)
    p.add_argument('--poly_degree', type=int, default=2)

    # System Specific Settings
    p.add_argument('--tMin', type=float, default=0.0)
    p.add_argument('--tMax', type=float, default=1.0)
    p.add_argument('--input_max', type=float, default=1.0)
    p.add_argument('--minWith', type=str, default='none', choices=['none', 'zero', 'target'])
    p.add_argument('--reachMode', type=str, default='forward', choices=['backward', 'forward'])
    p.add_argument('--reachAim', type=str, default='reach', choices=['avoid', 'reach'])
    p.add_argument('--setType', type=str, default='set', choices=['set', 'tube'])

    # Training Process Settings
    p.add_argument('--pretrain', action='store_true', default=True)
    p.add_argument('--pretrain_iters', type=int, default=2000)
    p.add_argument('--counter_start', type=int, default=0)
    p.add_argument('--counter_end', type=int, default=100e3)
    p.add_argument('--num_src_samples', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)

    # Verification Settings
    p.add_argument('--epsilon', type=float, default=0.35)
    p.add_argument('--epsilon_radius', type=float, default=0.1)
    p.add_argument('--max_iterations', type=int, default=5)
    
    # Add device argument
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                  help='Device to use for computation (cuda/cpu)')

    # Training Mode Settings
    p.add_argument('--quick_mode', action='store_true', default=False,
                  help='Enable quick testing mode with reduced epochs and iterations')
    p.add_argument('--full_mode', action='store_true', default=False,
                  help='Enable full training mode with complete epochs and iterations')
    
    args = p.parse_args()

    # Adjust parameters based on mode
    if args.quick_mode:
        args.num_epochs = 10
        args.max_iterations = 2
        args.epsilon = 0.35
    elif args.full_mode:
        args.num_epochs = 5000
        args.max_iterations = 10
        args.epsilon = 0.35

    return args

def cleanup():
    """Cleanup function to handle multiprocessing resources"""
    if hasattr(multiprocessing, '_ctx') and hasattr(multiprocessing._ctx, '_semaphore_tracker'):
        multiprocessing._ctx._semaphore_tracker.clear()

def load_model_safely(example, model_path, device):
    """Helper function to safely load model"""
    logger = logging.getLogger(__name__)
    try:
        # Use SingleBVPNet's load_checkpoint consistently
        model, checkpoint = example.model.load_checkpoint(model_path, device=device)
        example.model = model
        logger.info(f"Successfully loaded model from {model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

def find_best_model_path(model_dir):
    """Find the best available model path by checking the most recent numbered directory"""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(model_dir):
        logger.warning(f"Model directory {model_dir} does not exist")
        return None

    # Look in parent directory (./logs) for numbered directories
    parent_dir = os.path.dirname(model_dir)  # Gets ./logs from ./logs/double_integrator
    base_name = os.path.basename(model_dir)  # Gets 'double_integrator'
    
    # Find all numbered directories matching the base name
    numbered_dirs = []
    for d in os.listdir(parent_dir):
        full_path = os.path.join(parent_dir, d)
        if not os.path.isdir(full_path):
            continue
        # Match directories that start with base_name and end with a number
        if d.startswith(base_name) and '_' in d:
            try:
                num = int(d.split('_')[-1])
                numbered_dirs.append((num, d))
            except (ValueError, IndexError):
                continue
    
    # If we found numbered directories, use the highest one
    if numbered_dirs:
        _, latest_dir = max(numbered_dirs, key=lambda x: x[0])
        latest_path = os.path.join(parent_dir, latest_dir)
        checkpoint_dir = os.path.join(latest_path, 'checkpoints')
        
        if os.path.exists(checkpoint_dir):
            for model_file in ['model_final.pth', 'model_current.pth']:
                model_path = os.path.join(checkpoint_dir, model_file)
                if os.path.exists(model_path):
                    logger.info(f"Found model: {model_file} in numbered directory {latest_dir}")
                    return model_path
    
    # Only fall back to direct checkpoints directory if no numbered directories found
    direct_checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    if os.path.exists(direct_checkpoint_dir):
        logger.warning("Falling back to non-numbered directory structure")
        for model_file in ['model_final.pth', 'model_current.pth']:
            model_path = os.path.join(direct_checkpoint_dir, model_file)
            if os.path.exists(model_path):
                logger.info(f"Found model: {model_file} in direct checkpoints directory")
                return model_path
    
    logger.warning("No valid model file found in any directory")
    return None

def try_load_model_from_folder(example, folder_path, device, logger):
    """Try to load a model from a given folder path."""
    checkpoint_dir = os.path.join(folder_path, 'checkpoints')
    if os.path.exists(checkpoint_dir):
        for model_file in ['model_final.pth', 'model_current.pth']:
            model_path = os.path.join(checkpoint_dir, model_file)
            if os.path.exists(model_path):
                if load_model_safely(example, model_path, device):
                    logger.info(f"Loaded model from {model_path}")
                    return True
    return False

def main():
    # Register cleanup function
    atexit.register(cleanup)
    
    # Parse command line arguments
    args = parse_args()
    
    # Set up base logging without file handler initially
    configure_logging(None)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting experiment with example: {args.example}")
    logger.debug(f"Arguments: {args}")

    # Create example with explicit device
    device = torch.device(args.device)
    example = create_example(args.example, args)
    example.device = device
    logger.info(f"Using device: {device}")
    
    # Make sure base directory exists
    os.makedirs(args.logging_root, exist_ok=True)
    
    # Get appropriate experiment folder path and info
    exp_folder_path, _, prev_folder_path = get_experiment_folder(args.logging_root, args.example)
    loaded_model = False
    
    # Initialize components before loading model
    example.initialize_components()

    # Try to load model from previous experiment folder if it exists
    if prev_folder_path:
        loaded_model = try_load_model_from_folder(example, prev_folder_path, device, logger)
        if loaded_model:
            logger.info(f"Loaded model from previous experiment: {prev_folder_path}")

    # Set up new experiment folder and logging
    logger.info(f"Creating new experiment directory: {exp_folder_path}")
    setup_experiment_folder(exp_folder_path, create=True)
    log_file = os.path.join(exp_folder_path, 'training.log')
    configure_logging(log_file)
    example.root_path = exp_folder_path

    # Handle quick mode with loaded model
    if loaded_model and args.run_mode == 'train' and args.quick_mode:
        logger.info("Quick mode with existing model: Skipping training phase")
        args.run_mode = 'verify'

    # Run based on mode
    if args.run_mode == 'train':
        if not loaded_model:
            logger.info("Starting training with new model")
        else:
            logger.info("Starting training with loaded model")
        example.train()
    
    if args.run_mode == 'verify':
        logger.info("Starting verification phase")
        # Ensure model is initialized before verification
        if not hasattr(example, 'model') or example.model is None:
            example.initialize_components()
            
            # Try to load existing model
            model_dir = os.path.join(args.logging_root, args.example)
            model_path = find_best_model_path(model_dir)
            if model_path:
                if not load_model_safely(example, model_path, device):
                    logger.error("Failed to load model for verification. Please check model compatibility.")
                    return
            else:
                logger.error("No trained model found. Please train a model first.")
                return

        example.verify()
        # Plot results with current epsilon
        dreal_result_path = f"{example.root_path}/dreal_result.json"
        if os.path.exists(dreal_result_path):
            try:
                with open(dreal_result_path, 'r') as f:
                    dreal_result = json.load(f)
                    epsilon = dreal_result.get("epsilon", args.epsilon)
                logger.info(f"Using epsilon value: {epsilon} from dReal results")
                example.plot_final_model(example.model, example.root_path, epsilon)
            except Exception as e:
                logger.error(f"Error processing dReal results: {str(e)}")

    if args.run_mode == 'cegis':
        logger.info("Starting CEGIS phase")
        # Initialize model if needed
        if not hasattr(example, 'model') or example.model is None:
            example.initialize_components()
            model_dir = os.path.join(args.logging_root, args.example)
            model_path = find_best_model_path(model_dir)
            if model_path:
                if not load_model_safely(example, model_path, device):
                    logger.info("Failed to load existing model, creating new one")
            else:
                logger.info("No existing model found, creating new one")
            
            # Model will be used as is, whether newly initialized or loaded

        logger.info(f"Starting {'quick' if args.quick_mode else 'full'} CEGIS loop")
        logger.info(f"Parameters: epochs={args.num_epochs}, "
                   f"max_iterations={args.max_iterations}, "
                   f"epsilon={args.epsilon}")
        
        cegis = CEGISLoop(example, args)
        result = cegis.run()
        
        # Plot results after successful CEGIS
        example.plot_final_model(example.model, example.root_path, result.epsilon)
        
        if result.success:
            logger.info(f"CEGIS completed. Best epsilon: {result.epsilon}")
        else:
            logger.info(f"CEGIS failed to find a valid epsilon value. Best attempt: {result.epsilon}")

    logger.info("Experiment completed")

if __name__ == '__main__':
    main()
