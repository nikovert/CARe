import logging
from typing import Dict
from certreach.common.dataset import ReachabilityDataset

logger = logging.getLogger(__name__)

class Curriculum:
    """Manages curriculum learning progress for ReachabilityDataset."""
    def __init__(self, 
                 dataset: ReachabilityDataset,
                 total_steps: int,
                 time_min: float = 0.0,
                 time_max: float = 1.0,
                 rollout: bool = True):
        if not isinstance(dataset, ReachabilityDataset):
            raise TypeError(f"Dataset must be ReachabilityDataset, got {type(dataset)}")
            
        self.dataset = dataset
        self.total_steps = total_steps
        self.rollout = rollout
        self.time_min = time_min
        self.time_max = time_max
        self.current_step = 0
        self.is_pretraining = True if rollout else False
    
    def __len__(self):
        return self.total_steps

    def step(self, progress_flag: bool = False):
        """Update curriculum progress and dataset time range."""
        if progress_flag:
            if self.is_pretraining:
                logger.info("Pretraining finished, starting rollout.")
                self.is_pretraining = False
            self.current_step += 1
            t_min, t_max = self.get_time_range()
            self.dataset.update_time_range(t_min, t_max)
    
    def get_progress(self) -> float:
        """Get current curriculum progress."""
        if not self.rollout:
            return 1.0
        if self.is_pretraining:
            return 0.0
        else:
            progress = self.current_step/self.total_steps
            return min(progress, 1.0)

    def get_time_range(self) -> tuple[float, float]:
        """Get current time range based on curriculum progress."""
        progress = self.get_progress()
        current_max = self.time_min + (self.time_max - self.time_min) * progress
        return self.time_min, current_max
    
    def get_loss_weights(self, batch_size) -> Dict[str, float]:
        """
        Returns weights for different loss components based on current curriculum state.
        
        Returns:
            Dict[str, float]: Dictionary mapping loss names to their weights
        """
        
        if self.is_pretraining:
            # During pretraining, focus more on Dirichlet boundary conditions
            weights = {
                'dirichlet': 100,
                'diff_constraint_hom': 1.0
            }
        else:
            # Increase importance of homogeneous differential constraint
            weights = {
                'dirichlet': 10,
                'diff_constraint_hom': 1.0
            }
        
        return weights
