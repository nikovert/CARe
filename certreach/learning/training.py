import torch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
import time
import numpy as np
from pathlib import Path
import shutil
import logging
from typing import Callable, Optional
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
          pretrain_percentage: float = 0.01,  # Added curriculum parameters
          time_min: float = 0.0,
          time_max: float = 1.0,
          clip_grad: bool = True, 
          validation_fn: Optional[Callable] = None, 
          start_epoch: int = 0,
                    device: Optional[torch.device] = None,
          use_amp: bool = True,
          l1_lambda: float = 1e-2,  # Changed default to 1e-4 for L1 regularization
          weight_decay: float = 1e-2,  # Changed default to 1e-5 for L2 regularization
          **kwargs
          ) -> None:
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
        l1_lambda: L1 regularization strength
        weight_decay: L2 regularization strength
    
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
    
    # Enable automatic mixed precision for CUDA devices
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler() if use_amp else None  # Only initialize once
    
    # Enable CUDA optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Ensure model and data are on the correct device
    model = model.to(device)
    # Precompute zeros for L1 regularization to avoid repeated allocations
    l1_zeros = [torch.zeros_like(param, device=device) for param in model.parameters()]
    
    # Configure optimizer with device-specific settings
    optim = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        **kwargs
    )

    # Initialize curriculum
    curriculum = Curriculum(
        dataset=dataset,
        pretrain_percentage=pretrain_percentage,
        total_steps=epochs,
        time_min=time_min,
        time_max=time_max
    )

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
    checkpoints_dir = model_dir / 'checkpoints'

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    model.checkpoint_dir = checkpoints_dir  # Set checkpoint directory for model

    with tqdm(total=epochs) as pbar:
        train_losses = []
        for epoch in range(start_epoch, epochs): 
            start_time = time.time()
            # Update curriculum scheduler epoch at the start of each epoch
            curriculum.step()
            
            if epoch % epochs_til_checkpoint == 0 and epoch > 0:
                # Save periodic checkpoint using model's method
                model.save_checkpoint(
                    name='model_current',
                    optimizer=optim,
                    epoch=epoch
                )
                
                # Save losses separately for analysis
                np.savetxt(checkpoints_dir / f'train_losses_epoch_{epoch:04d}.txt',
                          np.array(train_losses))
                train_losses = []
                _, tmax = curriculum.get_time_range()
                if validation_fn is not None:
                    validation_fn(model, checkpoints_dir, epoch, tmax=tmax)

            # Get a fresh batch of data
            model_input, gt = dataset.get_batch()
            
            # Ensure coords requires gradients
            model_input['coords'].requires_grad_(True)
            
            optim.zero_grad(set_to_none=True)
            
            with torch.autocast(device.type, enabled=use_amp):
                model_output = model(model_input)
                losses = loss_fn(model_output, gt)
                
                # Get weights from curriculum
                batch_size = model_input['coords'].shape[0]  # Assuming first dimension is batch size
                loss_weights = curriculum.get_loss_weights(batch_size)
                
                # Apply weights to losses and normalize by batch size
                train_loss = sum(loss.mean() * loss_weights.get(name, 1.0)
                                for name, loss in losses.items())
                
                # Calculate total loss and add L1 regularization using precomputed zeros
                if l1_lambda > 0:
                    l1_loss = torch.tensor(0., device=device)
                    for param, zero in zip(model.parameters(), l1_zeros):
                        l1_loss += F.l1_loss(param, zero, reduction='sum')
                    train_loss += l1_lambda * l1_loss

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

            train_losses.append(train_loss.item())

            # Simplified progress reporting
            if epoch % max(100,(epochs //1000)) == 0:  # Report only 1000 times during training
                tqdm.write(f"Epoch {epoch}, Total Loss: {train_loss:.6f},"
                          f"L1 Reg: {(l1_lambda * l1_loss if l1_lambda > 0 else 0):.6f}, "
                          f"L2 Reg: {(weight_decay * sum((p ** 2).sum() for p in model.parameters())):.6f}, "
                          f"Time: {time.time() - start_time:.3f}s")
                curr_progress = curriculum.get_progress()
                tmin, tmax = curriculum.get_time_range()
                phase = "Pretraining" if curriculum.is_pretraining else "Curriculum"
                tqdm.write(f"{phase} - Progress: {curr_progress:.2%}, Time range: [{tmin:.3f}, {tmax:.3f}]")

            pbar.update(1)

        # Save final model
        model.save_checkpoint(
            name='model_final',
            optimizer=optim,
            epoch=epochs,
            training_completed=True
        )

        # Save final losses
        np.savetxt(checkpoints_dir / 'train_losses_final.txt', 
                   np.array(train_losses))
