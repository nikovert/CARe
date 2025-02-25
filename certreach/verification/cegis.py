import os
import time
import logging
import torch
from typing import Optional, List
from dataclasses import dataclass
from certreach.verification.SMT_verifier import SMTVerifier
from certreach.common.dataset import ReachabilityDataset
from certreach.learning.training import train

logger = logging.getLogger(__name__)

@dataclass
class TimingStats:
    """Data class to store timing information for each iteration"""
    training_time: float
    symbolic_time: float
    verification_time: float

@dataclass
class CEGISResult:
    """Data class to store CEGIS iteration results"""
    epsilon: float
    success: bool
    model_path: Optional[str] = None
    timing_history: Optional[List[TimingStats]] = None
    total_time: Optional[float] = None

class CEGISLoop:
    """Counter-Example Guided Inductive Synthesis (CEGIS) Loop implementation."""
    
    def __init__(self, example, args):
        self.example = example
        self.args = args # To pass on arguments to training and dataset creation
        self.device = torch.device(args.device)
        self.max_iterations = args.max_iterations
        self.current_epsilon = args.epsilon
        self.best_epsilon = float('inf')
        self.best_model_path = None
        self.best_model_state = None
        self.timing_history = []
        self.current_symbolic_model = None
        self.min_epsilon = args.min_epsilon
        self.pruning_percentage = getattr(args, 'pruning_percentage', 0.25)  # Default threshold if not specified
        self.prune_after_initial = getattr(args, 'prune_after_initial', True)  # Whether to prune after initial training
        
        # Initialize verifier - use the specified solver preference if available
        solver_preference = getattr(args, 'solver', 'auto')
        self.verifier = SMTVerifier(device=self.device, solver_preference=solver_preference)

        self.initial_lr = args.lr
        self.fine_tune_lr = args.lr * 0.1  # Lower learning rate for fine-tuning
        self.fine_tune_epochs = args.num_epochs // 4  # Fewer epochs for fine-tuning
                
        # Initialize dataset if not already existing
        self.dataset = ReachabilityDataset(
            batch_size=args.batch_size,
            tMin=args.tMin,
            tMax=args.tMax,
            seed=args.seed,
            device=self.device,  # Pass device explicitly
            num_states=example.NUM_STATES,  # Use the example's number of states
            compute_boundary_values=example.boundary_fn,
            percentage_in_counterexample=args.percentage_in_counterexample,
            percentage_at_t0=args.percentage_at_t0,
            epsilon_radius=args.epsilon_radius
        )
        
    def run(self, train_first = True) -> CEGISResult:
        """Run the CEGIS loop with proper CUDA memory management."""
        iteration_count = 0
        start_time = time.time()
        base_dir = self.example.root_path
        
        model_config = self.example.model.get_config()

        # Initial training if starting from scratch
        if train_first:
            # Create initial training directory
            initial_dir = os.path.join(base_dir, "initial_training")
            os.makedirs(initial_dir, exist_ok=True)
            self.example.root_path = initial_dir
            
            logger.info("Starting initial training before verification loop")
            train_start = time.time()
            train(
                model=self.example.model,
                dataset=self.dataset,
                epochs=self.args.num_epochs,
                lr=self.args.lr,
                epochs_til_checkpoint=self.args.epochs_til_ckpt,
                model_dir=self.example.root_path,
                loss_fn=self.example.loss_fn,
                pretrain_percentage=self.args.pretrain_percentage,
                time_min=self.args.tMin,
                time_max=self.args.tMax,
                validation_fn=self.example.validate,
                device=self.device,
                is_finetuning=False  # Initial training is not fine-tuning
            )
            self.last_training_time = time.time() - train_start
            
            # Prune the model after initial training if enabled
            if self.prune_after_initial:
                stats = self.example.model.prune_weights(self.pruning_percentage)
                logger.info(f"Pruning stats: {stats}")
                
                # Retrain the pruned model
                logger.info("Retraining pruned model")
                train_start = time.time()
                train(
                    model=self.example.model,
                    dataset=self.dataset,
                    epochs=self.args.num_epochs // 2,  # Shorter training period for fine-tuning
                    lr=self.args.lr * 0.5,  # Lower learning rate for fine-tuning
                    epochs_til_checkpoint=self.args.epochs_til_ckpt,
                    model_dir=self.example.root_path,
                    loss_fn=self.example.loss_fn,
                    pretrain_percentage=0.0,  # No pretraining needed
                    time_min=self.args.tMin,
                    time_max=self.args.tMax,
                    validation_fn=self.example.validate,
                    device=self.device,
                    is_finetuning=True  # Retraining pruned model is fine-tuning
                )
                self.last_training_time += time.time() - train_start

        while iteration_count < self.max_iterations:
            logger.info("Starting iteration %d with epsilon: %.4f", 
                       iteration_count + 1, self.current_epsilon)
            
            # Update system_specifics with current root_path before verification
            system_specifics = {
                'name': self.example.Name,
                'root_path': self.example.root_path,
                'reachMode': getattr(self.example, 'reachMode', 'forward'),
                'setType': getattr(self.example, 'setType', 'set'),
                'additional_constraints': getattr(self.example, 'additional_constraints', None)
            }
            
            # Extract model state and config, keeping tensors on CPU for verification
            with torch.no_grad():
                model_state = {k: v.cpu() for k, v in self.example.model.state_dict().items()}
            
            # Get verification result and timing info
            success, counterexample, timing_info = self.verifier.verify_system(
                model_state=model_state,
                model_config=model_config,
                system_specifics=system_specifics,  # Use updated system_specifics
                compute_hamiltonian=self.example.hamiltonian_fn,
                compute_boundary=self.example.boundary_fn,
                epsilon=self.current_epsilon
            )
            # Store timing information
            self.timing_history.append(TimingStats(
                training_time=getattr(self, 'last_training_time', 0.0),
                symbolic_time=timing_info['symbolic_time'],
                verification_time=timing_info['verification_time']
            ))
            
            if success:
                # Verification succeeded, try smaller epsilon
                with torch.no_grad():  # No gradients needed for model state saving
                    if self.current_epsilon < self.best_epsilon:
                        self.best_epsilon = self.current_epsilon
                        self.best_model_path = os.path.join(
                            self.example.root_path, "checkpoints", "model_final.pth"
                        )
                        self.best_model_state = {
                            k: v.clone() for k, v in self.example.model.state_dict().items()
                        }
                    # Check if we've reached minimum epsilon
                    if self.current_epsilon <= self.min_epsilon:
                        logger.info(f"Reached minimum epsilon threshold: {self.min_epsilon}")
                        break
                    self.current_epsilon *= 0.75  # Reduce epsilon by 25%
                    # Ensure we don't go below minimum epsilon
                    self.current_epsilon = max(self.current_epsilon, self.min_epsilon)
                    self.args.epsilon = self.current_epsilon
            else:
                # Train on counterexample
                self.current_epsilon *= 1.05  # Increase epsilon by 5%

                # Create a subdirectory for this counterexample iteration inside the checkpoint directory
                counter_dir = os.path.join(base_dir, f"iteration_{iteration_count+1}")
                os.makedirs(counter_dir, exist_ok=True)
                self.example.root_path = counter_dir
                logger.info(f"Created new iteration directory: {counter_dir}")

                train_start = time.time()
                self.dataset.add_counterexample(counterexample)
                
                # Fine-tune the model
                train(
                    model=self.example.model,
                    dataset=self.dataset,
                    epochs=self.fine_tune_epochs,
                    lr=self.fine_tune_lr,
                    epochs_til_checkpoint=self.args.epochs_til_ckpt,
                    model_dir=self.example.root_path,
                    loss_fn=self.example.loss_fn,
                    pretrain_percentage=0.0,  # No pretraining during fine-tuning
                    time_min=self.args.tMin,
                    time_max=self.args.tMax,
                    validation_fn=self.example.validate,
                    device=self.device,
                    is_finetuning=True,  # Set fine-tuning flag for counterexample training
                    momentum=0.9  # Add momentum for fine-tuning
                )
                self.last_training_time = time.time() - train_start
                self.current_symbolic_model = None  # Reset symbolic model after training

            iteration_count += 1
        
        total_time = time.time() - start_time
        return self._finalize_results(total_time)

    def _finalize_results(self, total_time: float) -> CEGISResult:
        """Restore best model and generate final visualizations."""
        if not self.best_model_state:
            return CEGISResult(
                epsilon=self.best_epsilon,
                success=False,
                timing_history=self.timing_history,
                total_time=total_time
            )

        logger.info("Best verification achieved with epsilon: %.4f", self.best_epsilon)
        logger.info("Timing Statistics:")
        logger.info("Total time: %.2f seconds", total_time)
        
        # Log timing statistics for each iteration
        for i, stats in enumerate(self.timing_history, 1):
            logger.debug("Iteration %d:", i)
            logger.debug("  Training time: %.2fs", stats.training_time)
            logger.debug("  Symbolic model generation: %.2fs", stats.symbolic_time)
            logger.debug("  dReal verification: %.2fs", stats.verification_time)
        
        self.example.model.cpu()
        self.example.model.load_state_dict(
            {k: v.cpu() for k, v in self.best_model_state.items()}
        )
        self.example.model.to(self.device)
        
        self.example.plot_final_model(
            self.example.model,
            self.example.root_path,
            self.best_epsilon,
            save_file="Final_Best_Model_Comparison.png"
        )
        
        # Include pruning information in the final model stats if available
        if (hasattr(self.example.model, 'is_pruned') and 
            hasattr(self.example.model, 'get_pruning_statistics') and 
            self.example.model.is_pruned):
            pruning_stats = self.example.model.get_pruning_statistics()
            logger.info("Final pruning statistics:")
            logger.info(f"Pruning ratio: {pruning_stats['pruning_ratio']:.2%}")
            logger.info(f"Pruned parameters: {pruning_stats['pruned_params']}")
            logger.info(f"Total parameters: {pruning_stats['total_params']}")
        
        return CEGISResult(
            epsilon=self.best_epsilon,
            success=True,
            model_path=self.best_model_path,
            timing_history=self.timing_history,
            total_time=total_time
        )
