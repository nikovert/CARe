from typing import Dict, Any, Optional
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from pathlib import Path
import logging

from .networks import NetworkConfig, SingleBVPNet

class ModelTuner:
    def __init__(self, 
                 base_config: NetworkConfig,
                 num_samples: int = 10,
                 max_epochs: int = 100,
                 gpus_per_trial: float = 0.5,
                 cpu_per_trial: int = 2):
        """
        Initialize the model tuner.
        
        Args:
            base_config: Base network configuration
            num_samples: Number of trials to run
            max_epochs: Maximum epochs per trial
            gpus_per_trial: GPUs per trial (fractional values allowed)
            cpu_per_trial: CPUs per trial
        """
        self.base_config = base_config
        self.num_samples = num_samples
        self.max_epochs = max_epochs
        self.gpus_per_trial = gpus_per_trial
        self.cpu_per_trial = cpu_per_trial
        
    def train_model(self, config: Dict[str, Any], checkpoint_dir: Optional[str] = None):
        """Training function for Ray Tune."""
        # Merge base config with tuning config
        model_config = {**self.base_config.to_dict(), **config}
        model = SingleBVPNet(model_config)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
        
        # Load checkpoint if exists
        if checkpoint_dir:
            checkpoint = Path(checkpoint_dir) / "checkpoint.pth"
            model, checkpoint_dict = SingleBVPNet.load_checkpoint(checkpoint, device)
            optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        
        # Training loop
        for epoch in range(self.max_epochs):
            # Your training logic here
            loss = self._train_epoch(model, optimizer)
            
            # Report metrics to Ray Tune
            tune.report(loss=loss)
            
            # Save checkpoint
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = Path(checkpoint_dir) / "checkpoint.pth"
                model.save_checkpoint(path, optimizer=optimizer, epoch=epoch)
    
    def _train_epoch(self, model, optimizer):
        """Implement your training logic here."""
        # Placeholder for actual training logic
        raise NotImplementedError("Implement your training logic here")
    
    def tune_model(self, search_alg: Optional[str] = "bayesopt"):
        """
        Run hyperparameter tuning.
        
        Args:
            search_alg: Search algorithm ('bayesopt' or None for random search)
        """
        # Get search space from NetworkConfig
        search_space = NetworkConfig.get_tune_config()
        
        # Add training-specific parameters
        search_space.update({
            "lr": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([32, 64, 128, 256])
        })
        
        # Configure search algorithm
        if search_alg == "bayesopt":
            search_algorithm = BayesOptSearch(
                metric="loss",
                mode="min",
                random_search_steps=10
            )
        else:
            search_algorithm = None
        
        # Configure scheduler
        scheduler = ASHAScheduler(
            time_attr='training_iteration',
            metric='loss',
            mode='min',
            max_t=self.max_epochs,
            grace_period=10,
            reduction_factor=2
        )
        
        # Run tuning
        analysis = tune.run(
            self.train_model,
            config=search_space,
            search_alg=search_algorithm,
            scheduler=scheduler,
            num_samples=self.num_samples,
            resources_per_trial={
                "cpu": self.cpu_per_trial,
                "gpu": self.gpus_per_trial
            },
            local_dir="./ray_results",
            name="hjreach_tune",
            log_to_file=True
        )
        
        # Get best trial
        best_trial = analysis.best_trial
        logging.info(f"Best trial config: {best_trial.config}")
        logging.info(f"Best trial final loss: {best_trial.last_result['loss']}")
        
        return analysis
