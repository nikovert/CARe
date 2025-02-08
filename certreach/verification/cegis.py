import os
import json
import time
import logging
import torch
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from certreach.verification.verify import verify_system
from copy import deepcopy

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
        self.current_symbolic_model = None
        
    @torch.no_grad()
    def run(self) -> CEGISResult:
        """Run the CEGIS loop with proper CUDA memory management."""
        iteration_count = 0
        start_time = time.time()
        
        while iteration_count < self.max_iterations:
            logger.info(f"Starting iteration {iteration_count + 1} with epsilon: {self.current_epsilon}")
            
            self.example.train()

            # Create CPU clone with same architecture as original model
            with torch.no_grad():
                cpu_model = deepcopy(self.example.model)  # This preserves architecture
                cpu_model.cpu()  # Move the clone to CPU
            
            # Get verification result and timing info
            verification_result, timing_info, symbolic_model = verify_system(
                model=cpu_model,
                root_path=self.example.root_path,
                system_type=self.example.Name,
                epsilon=self.current_epsilon,
                verification_fn=self.example.verification_fn,
                symbolic_model=self.current_symbolic_model
            )
            
            # Store symbolic model for potential reuse
            self.current_symbolic_model = symbolic_model
            
            # Store timing information
            self.timing_history.append(TimingStats(
                training_time=getattr(self.example, 'last_training_time', 0.0),
                symbolic_time=timing_info['symbolic_time'],
                verification_time=timing_info['verification_time']
            ))
            
            counterexample = self._process_verification_results()
            
            if counterexample is None:
                # Verification succeeded, try smaller epsilon
                if self.current_epsilon < self.best_epsilon:
                    self.best_epsilon = self.current_epsilon
                    self.best_model_path = os.path.join(
                        self.example.root_path, "checkpoints", "model_final.pth"
                    )
                    self.best_model_state = {
                        k: v.clone() for k, v in self.example.model.state_dict().items()
                    }
                self.current_epsilon *= 0.5
                self.args.epsilon = self.current_epsilon
            else:
                # Train on counterexample
                train_start = time.time()
                counterexample.requires_grad=True
                self.example.train(counterexample=counterexample)
                train_time = time.time() - train_start
                self.example.last_training_time = train_time
                self.current_symbolic_model = None  # Reset symbolic model after training
                
            iteration_count += 1
        
        total_time = time.time() - start_time
        return self._finalize_results(total_time)
    
    def _process_verification_results(self) -> Optional[torch.Tensor]:
        """Process verification results and return counterexample if found."""
        dreal_result_path = f"{self.example.root_path}/dreal_result.json"
        with open(dreal_result_path, 'r') as f:
            result = json.load(f)
            
        if "HJB Equation Satisfied" in result["result"]:
            logger.info("HJB Equation satisfied. Reducing epsilon.")
            return None
            
        logger.info("Counterexample found. Will retrain model.")
        # Extract counterexample points from the result string
        counterexample_points = []
        if "result" in result:
            lines = result["result"].split('\n')
            for line in lines:
                if line.startswith('x_'):
                    interval_str = line.split(':')[1].strip()[1:-1]
                    lower, upper = map(float, interval_str.split(','))
                    point = (lower + upper) / 2
                    counterexample_points.append(point)
        
        if not counterexample_points:
            logger.warning("No counterexample points found in dReal result")
            return None
        
        return torch.tensor(counterexample_points, device=self.device)
    
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
