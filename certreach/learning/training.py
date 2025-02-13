import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
from pathlib import Path
import shutil
import logging
from typing import Callable, Optional, Dict
from torch.cuda.amp import autocast, GradScaler
from .curriculum import Curriculum
from ..common.dataset import ReachabilityDataset

logger = logging.getLogger(__name__)

def train(model: torch.nn.Module, 
          dataset: ReachabilityDataset,  # Updated type hint
          epochs: int, 
          lr: float, 
          epochs_til_checkpoint: int, 
          model_dir: str, 
          loss_fn: Callable, 
          pretrain_percentage: float = 0.1,  # Added curriculum parameters
          time_min: float = 0.0,
          time_max: float = 1.0,
          clip_grad: bool = False, 
          loss_schedules: Optional[Dict[str, Callable]] = None, 
          validation_fn: Optional[Callable] = None, 
          start_epoch: int = 0,
                    device: Optional[torch.device] = None,
          use_amp: bool = True) -> None:
    """
    Train a model using curriculum learning for reachability problems.
    
    Args:
        model: Neural network model to train
        dataset: Dataset for training, must be a ReachabilityDataset instance
        epochs: Total number of training epochs
        lr: Learning rate for the Adam optimizer
        epochs_til_checkpoint: Number of epochs between saving checkpoints
        model_dir: Directory to save model checkpoints and tensorboard logs
        loss_fn: Loss function that takes model output and ground truth as input
        pretrain_percentage: Fraction of total epochs to spend in pretraining phase (0 to 1)
        time_min: Minimum time value for curriculum learning
        time_max: Maximum time value for curriculum learning
        clip_grad: Whether to clip gradients during training
        loss_schedules: Dictionary of callable schedules for each loss component
        validation_fn: Optional function to run validation during checkpoints
        start_epoch: Epoch to start or resume training from
        device: Device to use for training (default: CUDA if available, else CPU)
        use_amp: Whether to use automatic mixed precision training
    
    Raises:
        TypeError: If dataset is not an instance of ReachabilityDataset
    
    Notes:
        - Uses curriculum learning with two phases:
          1. Pretraining phase: Trains on a subset of time values
          2. Curriculum phase: Gradually increases the time range
        - Saves checkpoints periodically and logs metrics to tensorboard
        - Supports automatic mixed precision training on CUDA devices
    """
    if not isinstance(dataset, ReachabilityDataset):
        raise TypeError(f"Dataset must be an instance of ReachabilityDataset, got {type(dataset)}")

    # Use provided device or default to CUDA if available
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize optimizer first as it's needed for curriculum scheduler
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize curriculum
    curriculum = Curriculum(
        dataset=dataset,
        pretrain_percentage=pretrain_percentage,
        total_steps=epochs,
        time_min=time_min,
        time_max=time_max
    )

    # Initialize gradient scaler for AMP
    scaler = torch.GradScaler() if use_amp and device.type == 'cuda' else None

    # Load the checkpoint if required
    if start_epoch > 0:
        try:
            model_path = Path(model_dir) / 'checkpoints' / f'model_epoch_{start_epoch:04d}.pth'
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model'])
            model.train()
            optim.load_state_dict(checkpoint['optimizer'])
            logger.info(f"Loaded checkpoint from epoch {start_epoch}")
        except FileNotFoundError:
            logger.error(f"Checkpoint file not found: {model_path}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")

    else:
        # Start training from scratch
        if Path(model_dir).exists():
            logger.info(f"The model directory {model_dir} exists. Deleting and recreating...")
            shutil.rmtree(model_dir)
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Make sure all path operations use Path consistently
    model_dir = Path(model_dir)
    summaries_dir = model_dir / 'summaries'
    checkpoints_dir = model_dir / 'checkpoints'

    summaries_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    model.checkpoint_dir = checkpoints_dir  # Set checkpoint directory for model

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    steps_til_summary = int(epochs/1000)
    with tqdm(total=epochs) as pbar:
        train_losses = []
        for epoch in range(start_epoch, epochs):
            start_time = time.time()
            # Update curriculum scheduler epoch at the start of each epoch
            curriculum.step()
            
            if epoch % epochs_til_checkpoint == 0 and epoch > 0:
                # Save periodic checkpoint using model's method
                model.save_checkpoint(
                    name=f'model_epoch_{epoch:04d}',
                    optimizer=optim,
                    epoch=epoch,
                    train_losses=train_losses
                )
                
                # Save losses separately for analysis
                np.savetxt(checkpoints_dir / f'train_losses_epoch_{epoch:04d}.txt',
                          np.array(train_losses))

                if validation_fn is not None:
                    validation_fn(model, checkpoints_dir, epoch)

            # Get a fresh batch of data
            model_input, gt = dataset.get_batch()

            
            optim.zero_grad(set_to_none=True)
            
            with torch.autocast(device.type, enabled=use_amp):
                model_output = model(model_input)
                losses = loss_fn(model_output, gt)
                
                # Get weights from curriculum
                loss_weights = curriculum.get_loss_weights()
                
                # Apply weights to losses and normalize by batch size
                batch_size = model_input['coords'].shape[0]  # Assuming first dimension is batch size
                weighted_losses = {
                    name: (1000*loss.mean() / batch_size) * loss_weights.get(name, 1.0)
                    for name, loss in losses.items()
                }
                
                train_loss = sum(weighted_losses.values())

            # Log both raw and weighted losses
            for loss_name, loss in losses.items():
                raw_loss = 1000*loss.mean() / batch_size  # Normalize raw loss
                weight = loss_weights.get(loss_name, 1.0)
                weighted_loss = raw_loss * weight
                
                writer.add_scalar(f"{loss_name}/raw", raw_loss, total_steps)
                writer.add_scalar(f"{loss_name}/weight", weight, total_steps)
                writer.add_scalar(f"{loss_name}/weighted", weighted_loss, total_steps)
                
                if loss_schedules and loss_name in loss_schedules:
                    schedule_weight = loss_schedules[loss_name](total_steps)
                    writer.add_scalar(f"{loss_name}/schedule_weight", schedule_weight, total_steps)
                    weighted_loss *= schedule_weight

            train_losses.append(train_loss.item())
            writer.add_scalar("total_train_loss", train_loss, total_steps)

            if scaler is not None:
                scaler.scale(train_loss).backward()
                if clip_grad:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optim)
                scaler.update()
            else:
                train_loss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()

            if total_steps % steps_til_summary == 0:
                tqdm.write(f"Epoch {epoch}, Total loss {train_loss:.6f}, "
                            f"iteration time {time.time() - start_time:.6f}")
                curr_progress = curriculum.get_progress()
                tmin, tmax = curriculum.get_time_range()
                phase = "Pretraining" if curriculum.is_pretraining else "Curriculum"
                tqdm.write(f"{phase} phase - Progress: {curr_progress:.2%}, Time range: [{tmin:.3f}, {tmax:.3f}]")

                # Add curriculum progress to tensorboard
                writer.add_scalar("curriculum/progress", curr_progress, total_steps)
                writer.add_scalar("curriculum/time_max", tmax, total_steps)

            total_steps += 1
            pbar.update(1)

        # Save final model
        model.save_checkpoint(
            name='model_final',
            optimizer=optim,
            epoch=epochs,
            train_losses=train_losses,
            training_completed=True
        )

        # Save final losses
        np.savetxt(checkpoints_dir / 'train_losses_final.txt', 
                   np.array(train_losses))
