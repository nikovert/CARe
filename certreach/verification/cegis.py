import os
import json
import time
import logging
import torch
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from certreach.verification.verify import verify_system

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
        self.args = args
        self.device = torch.device(args.device)
        self.max_iterations = args.max_iterations
        self.current_epsilon = args.epsilon
        self.best_epsilon = float('inf')
        self.best_model_path = None
        self.best_model_state = None
        self.timing_history = []
        
    @torch.no_grad()
    def run(self) -> CEGISResult:
        """Run the CEGIS loop with proper CUDA memory management."""
        iteration_count = 0
        start_time = time.time()
        
        while iteration_count < self.max_iterations:
            logger.info(f"Starting iteration {iteration_count + 1} with epsilon: {self.current_epsilon}")
            
            # Get verification result and timing info
            verification_result, timing_info = verify_system(
                model=self.example.model,
                root_path=self.example.root_path,
                system_type=self.example.Name,
                epsilon=self.current_epsilon
            )
            
            # Store timing information
            self.timing_history.append(TimingStats(
                training_time=getattr(self.example, 'last_training_time', 0.0),
                symbolic_time=timing_info['symbolic_time'],
                verification_time=timing_info['verification_time']
            ))
            
            result = self._process_verification_results()
            
            if not result.success:
                break
                
            iteration_count += 1
        
        total_time = time.time() - start_time
        return self._finalize_results(total_time)
    
    def _process_verification_results(self) -> CEGISResult:
        """Process verification results."""
        dreal_result_path = f"{self.example.root_path}/dreal_result.json"
        with open(dreal_result_path, 'r') as f:
            result = json.load(f)
            
        if "HJB Equation Satisfied" in result["result"]:
            return self._handle_success()
        else:
            return self._handle_counterexample(result)
    
    def _handle_success(self) -> CEGISResult:
        """Handle successful verification with model state preservation."""
        logger.info("HJB Equation satisfied. Reducing epsilon.")
        
        if self.current_epsilon < self.best_epsilon:
            self.best_epsilon = self.current_epsilon
            self.best_model_path = os.path.join(
                self.example.root_path, "checkpoints", "model_final.pth"
            )
            self.best_model_state = {
                k: v.clone() for k, v in self.example.model.state_dict().items()
            }
        
        self.current_epsilon *= 0.9
        self.args.epsilon = self.current_epsilon
        return CEGISResult(epsilon=self.current_epsilon, success=True)
    
    def _handle_counterexample(self, result: Dict[str, Any]) -> CEGISResult:
        """Handle counterexample with efficient device management."""
        logger.info("Counterexample found. Retraining model.")
        
        # Time the training process
        train_start = time.time()
        self.example.train(counterexample=True)
        train_time = time.time() - train_start
        
        # Store training time for this iteration
        self.example.last_training_time = train_time
        
        return CEGISResult(epsilon=self.current_epsilon, success=True)
    
    def _finalize_results(self, total_time: float) -> CEGISResult:
        """Restore best model and generate final visualizations."""
        if not self.best_model_state:
            return CEGISResult(
                epsilon=self.best_epsilon,
                success=False,
                timing_history=self.timing_history,
                total_time=total_time
            )

        logger.info(f"Best verification achieved with epsilon: {self.best_epsilon}")
        logger.info("\nTiming Statistics:")
        logger.info(f"Total time: {total_time:.2f} seconds")
        
        # Log timing statistics for each iteration
        for i, stats in enumerate(self.timing_history, 1):
            logger.info(f"\nIteration {i}:")
            logger.info(f"  Training time: {stats.training_time:.2f}s")
            logger.info(f"  Symbolic model generation: {stats.symbolic_time:.2f}s")
            logger.info(f"  dReal verification: {stats.verification_time:.2f}s")
        
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
        
        return CEGISResult(
            epsilon=self.best_epsilon,
            success=True,
            model_path=self.best_model_path,
            timing_history=self.timing_history,
            total_time=total_time
        )
