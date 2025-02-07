import configargparse
from examples.factories import create_example, EXAMPLE_NAMES
import json
import os
import atexit
import multiprocessing
import logging
from examples.log import configure_logging
import torch
from certreach.verification.cegis import CEGISLoop

def parse_args():
    """Parse command line arguments."""
    p = configargparse.ArgumentParser(description='Train and verify neural networks for reachability analysis')
    
    # Required arguments
    p.add_argument('--example', type=str, required=True, choices=EXAMPLE_NAMES,
                       help='Name of the example to run')
    p.add_argument('--run_mode', type=str, required=True, choices=['train', 'verify', 'all'],
                       help='Mode to run: train, verify, or all (both)')
    
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
    p.add_argument('--use_polynomial', action='store_true', default=True)
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
    p.add_argument('--iterate', type=bool, default=False)
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
        args.epsilon = 0.5
        args.iterate = True
    elif args.full_mode:
        args.num_epochs = 30000
        args.max_iterations = 10
        args.epsilon = 0.35
        args.iterate = True

    return args

def cleanup():
    """Cleanup function to handle multiprocessing resources"""
    if hasattr(multiprocessing, '_ctx') and hasattr(multiprocessing._ctx, '_semaphore_tracker'):
        multiprocessing._ctx._semaphore_tracker.clear()

def main():
    # Register cleanup function
    atexit.register(cleanup)
    
    # Parse command line arguments
    args = parse_args()
    
    # Configure logging at application startup
    log_file = os.path.join(args.logging_root, args.example, 'training.log')
    configure_logging(log_file)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting experiment with example: {args.example}")
    logger.debug(f"Arguments: {args}")

    # Create example with explicit device
    device = torch.device(args.device)
    example = create_example(args.example, args)
    example.device = device
    logger.info(f"Using device: {device}")
    
    # Run based on mode
    if args.run_mode in ['train', 'all']:
        logger.info("Starting training phase")
        example.train()
    
    if args.run_mode in ['verify', 'all']:
        logger.info("Starting verification phase")
        if args.iterate:
            logger.info(f"Starting {'quick' if args.quick_mode else 'full'} CEGIS loop")
            logger.info(f"Parameters: epochs={args.num_epochs}, "
                       f"max_iterations={args.max_iterations}, "
                       f"epsilon={args.epsilon}")
            
            cegis = CEGISLoop(example, args)
            result = cegis.run()
            
            if result.success:
                logger.info(f"CEGIS completed. Best epsilon: {result.epsilon}")
            else:
                logger.error(f"CEGIS failed: {result.error_msg}")
        else:
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

    logger.info("Experiment completed")

if __name__ == '__main__':
    main()
