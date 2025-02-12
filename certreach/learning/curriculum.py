import logging
from typing import Dict
from ..common.dataset import ReachabilityDataset

logger = logging.getLogger(__name__)

class Curriculum:
    """Manages curriculum learning progress for ReachabilityDataset."""
    
    def __init__(self, 
                 dataset: ReachabilityDataset,
                 pretrain_percentage: int,
                 total_steps: int,
                 time_min: float = 0.0,
                 time_max: float = 1.0):
        if not isinstance(dataset, ReachabilityDataset):
            raise TypeError(f"Dataset must be ReachabilityDataset, got {type(dataset)}")
            
        self.dataset = dataset
        self.pretrain_percentage = pretrain_percentage
        self.total_steps = total_steps
        self.time_min = time_min
        self.time_max = time_max
        self.current_step = 0
    
    def __len__(self):
        return self.total_steps

    def step(self):
        """Update curriculum progress and dataset time range."""
        self.current_step += 1
        tmin, tmax = self.get_time_range()
        self.dataset.update_time_range(tmin, tmax)
    
    def get_progress(self) -> float:
        """Get current curriculum progress."""
        if self.current_step < self.pretrain_percentage*self.total_steps:
            return 0.0
        else:
            progress = (self.current_step/self.total_steps - self.pretrain_percentage)/(1 - self.pretrain_percentage)
            return min(progress, 1.0)

    def get_time_range(self) -> tuple[float, float]:
        """Get current time range based on curriculum progress."""
        progress = self.get_progress()
        current_max = self.time_min + (self.time_max - self.time_min) * progress
        return self.time_min, current_max
            
    @property
    def is_pretraining(self) -> bool:
        return not (self.get_progress() > 0.0)
    
    def get_loss_weights(self) -> Dict[str, float]:
        """
        Returns weights for different loss components based on current curriculum state.
        
        Returns:
            Dict[str, float]: Dictionary mapping loss names to their weights
        """
        progress = self.get_progress()
        
        if self.is_pretraining:
            # During pretraining, focus more on Dirichlet boundary conditions
            weights = {
                'dirichlet': 1.0,
                'diff_constraint_hom': 0.0
            }
        else:
            # Gradually increase importance of homogeneous differential constraint
            weights = {
                'dirichlet': 1.0,
                'diff_constraint_hom': min(1.0, 0.5 + 0.5 * progress)
            }
        
        return weights
