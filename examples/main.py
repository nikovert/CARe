import configargparse
import os
import logging
from examples.log import configure_logging
import torch
from care.verification.cegis import CEGISLoop
from examples.factories import create_example, EXAMPLE_NAMES
from examples.experiment_utils import get_experiment_folder, setup_experiment_folder
import numpy as np
import random  

# Create module-level logger
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    p = configargparse.ArgumentParser(description='Train and verify neural networks for reachability analysis')
    
    # Required arguments
    p.add_argument('--example', type=str, required=True, choices=EXAMPLE_NAMES,
                  help='Name of the example to run')
    p.add_argument('--run_mode', type=str, required=True, choices=['train', 'cegis'],
                  help='Mode to run: train, verify, or cegis')
    
    # Logging and Experiment Settings
    p.add_argument('--logging_root', type=str, default='./logs',
                  help='Root directory for logging')
    p.add_argument('--load_model', action='store_true', default=False,
                  help='Whether to load model from previous experiment')

    # Training Settings
    p.add_argument('--batch_size', type=int, default=85000,
                  help='Number of points to sample per batch')
    p.add_argument('--lr', type=float, default=1e-4,
                  help='Learning rate for optimizer')
    p.add_argument('--num_epochs', type=int, default=int(1e5),
                  help='Number of training epochs for curriculum training (1/10th of max iterations)')
    p.add_argument('--epochs_til_ckpt', type=int, default=int(5000),
                  help='Number of epochs between model checkpoints')

    # Model Settings
    p.add_argument('--model_type', type=str, default='sine', 
                  choices=['sine', 'relu', 'relu_primitive'],
                  help='Activation function for the neural network')
    p.add_argument('--num_hl', type=int, default=0,
                  help='Number of hidden layers')
    p.add_argument('--num_nl', type=int, default=64,
                  help='Number of neurons per layer')
    p.add_argument('--use_polynomial', action='store_true', default=False,
                  help='Whether to use polynomial features')
    p.add_argument('--poly_degree', type=int, default=2,
                  help='Degree of polynomial features if used')

    # System Specific Settings
    p.add_argument('--t_min', type=float, default=0.0,
                  help='Minimum time for reachability analysis')
    p.add_argument('--t_max', type=float, default=1.0,
                  help='Maximum time for reachability analysis')
    p.add_argument('--input_max', type=float, default=1.0,
                  help='Maximum input magnitude for system')
    p.add_argument('--min_with', type=str, default='none', choices=['none', 'target'],
                  help='Type of minimum operation to use in HJ reachability')
    p.add_argument('--reach_mode', type=str, default='forward', choices=['backward', 'forward'],
                  help='Direction of reachability computation')
    p.add_argument('--reach_aim', type=str, default='reach', choices=['avoid', 'reach'],
                  help='Whether to compute reach or avoid sets')
    p.add_argument('--set_type', type=str, default='set', choices=['set', 'tube'],
                  help='Whether to compute reachable set or tube')

    # Training Process Settings
    p.add_argument('--prune_after_initial', action='store_true', default=False,
                  help='Whether to prune the network after initial training phase')
    p.add_argument('--seed', type=int, default=0,
                  help='Random seed for reproducibility')

    # Verification Settings
    p.add_argument('--epsilon', type=float, default=0.35,
                  help='Initial epsilon for verification')
    p.add_argument('--min_epsilon', type=float, default=0.01,
                  help='Minimum epsilon to achieve before terminating CEGIS')
    p.add_argument('--epsilon_radius', type=float, default=0.1,
                  help='Radius around counterexample points for sampling')
    p.add_argument('--max_iterations', type=int, default=7,
                  help='Maximum number of CEGIS iterations')
    p.add_argument('--solver', type=str, default='dreal', choices=['dreal', 'z3', 'marabou'],
                  help='SMT solver to use for verification')
    
    # Add device argument
    p.add_argument('--device', type=str, default=None,
                  help='Device to use for computation (cuda/cpu)')
    
    # Add solution checking argument
    p.add_argument('--check_solution', action='store_true', default=False,
                  help='Compare results with true values after verification')

    # Dataset Settings
    p.add_argument('--percentage_in_counterexample', type=float, default=5.0,
                  help='Percentage of points to sample near counterexamples')
    p.add_argument('--percentage_at_t0', type=float, default=2.0,
                  help='Percentage of points to sample at t=0')

    args = p.parse_args()

    # Set device default if not explicitly set
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set pin_memory based on device type if not explicitly set
    args.pin_memory = args.device == 'cpu'

    return args

def cleanup():
    """Simple cleanup function"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model_from_folder(example, folder_path):
    """Try to load a model from a given folder path."""
    checkpoint_dir = os.path.join(folder_path, 'checkpoints')
    if os.path.exists(checkpoint_dir):
        for model_file in ['model_final.pth', 'model_current.pth']:
            model_path = os.path.join(checkpoint_dir, model_file)
            if os.path.exists(model_path):
                try:
                    example.model.load_checkpoint(model_path, example.device)
                    logger.info(f"Loaded model from {model_path}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to load model from {model_path}: {e}")
                    continue
    return False

def main():
    """Main function to run the experiments."""
    args = parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Make sure base directory exists
    os.makedirs(args.logging_root, exist_ok=True)
    
    # Get appropriate experiment folder path and info
    exp_folder_path, _, prev_folder_path = get_experiment_folder(args.logging_root, args.example)
    setup_experiment_folder(exp_folder_path, create=True)
    
    # Configure logging - this sets up the root logger that all module loggers inherit from
    log_file = os.path.join(exp_folder_path, 'training.log')
    configure_logging(log_file, log_level=logging.DEBUG)
    
    # Log the choice of arguments
    logger.info(f"Arguments: {args.__dict__}")
    
    # Simplified device setup for single GPU
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda:0')  # Use the only GPU
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    logger.info(f"Starting experiment with example: {args.example}")

    # Create example with explicit device
    example = create_example(args.example, args)
    example.device = device  # Ensure model is on device
    
    # Initialize components before loading model
    example.initialize_components()

    # Try to load model from previous experiment folder if it exists
    loaded_model = False
    if prev_folder_path and args.load_model:
        loaded_model = load_model_from_folder(example, prev_folder_path)

    # Print model information to logger
    logger.info("Model Architecture:")
    logger.info(example.model)
    
    example.root_path = exp_folder_path

    # Run based on mode
    if args.run_mode == 'train':
        logger.info("Starting training with " + ("loaded" if loaded_model else "new") + " model")
        example.train()
            
        # Add comparison with true values if requested
        if args.check_solution:
            logger.info("Comparing results with true values...")
            example.compare_with_true_values()
    elif args.run_mode == 'cegis':
        logger.info("Starting CEGIS phase")
        
        cegis = CEGISLoop(example, args)
        result = cegis.run(train_first=not loaded_model)
        
        example.plot_final_model(example.model, example.root_path, result.epsilon)
        logger.info(f"CEGIS {'completed' if result.success else 'failed'}. "
                f"Best epsilon: {result.epsilon}")
        
        # Add comparison with true values if requested
        if args.check_solution:
            logger.info("Comparing results with true values...")
            try:
                example.compare_with_true_values()
            except Exception as e:
                logger.error(f"Error comparing with true values: {e}")

    logger.info("Experiment completed")
    cleanup()

if __name__ == '__main__':
    main()
