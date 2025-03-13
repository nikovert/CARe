import torch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
import time
import numpy as np
from pathlib import Path
import logging
from typing import Callable, Optional
from .curriculum import Curriculum
from ..common.dataset import ReachabilityDataset

logger = logging.getLogger(__name__)

def train(model: torch.nn.Module, 
          dataset: ReachabilityDataset,
          max_epochs: int, 
          model_dir: str, 
          loss_fn: Callable, 
          epochs_til_checkpoint: int = 5000, 
          curriculum_epochs: int = 0,
          lr: float = 1e-4, 
          time_min: float = 0.0,
          time_max: float = 1.0,
          validation_fn: Optional[Callable] = None, 
          epsilon: float = 0.2,
          device: Optional[torch.device] = None,
          clip_grad: bool = True, 
          use_amp: bool = True,
          l1_lambda: float = 1e-5,
          weight_decay: float = 1e-5,
          is_finetuning: bool = False,
          momentum: float = 0.9,
          **kwargs
          ) -> None:
    """
    Train a model using curriculum learning for reachability problems.
    
    Args:
        model: Neural network model to train
        dataset: Dataset for training, must be a ReachabilityDataset instance
        max_epochs: Maximum number of training epochs
        curriculum_epochs: Number of epochs to run the curriculum learning for
        lr: Learning rate for the optimizer
        epochs_til_checkpoint: Number of epochs between saving checkpoints
        model_dir: Directory to save model checkpoints and logs
        loss_fn: Loss function that takes model output and ground truth as input
        pretrain_percentage: Fraction of curriculum epochs to spend in pretraining phase (0 to 1)
        time_min: Minimum time value for curriculum learning
        time_max: Maximum time value for curriculum learning
        validation_fn: Optional function to run validation during checkpoints
        epsilon: Stopping criterion threshold - stops if loss falls below this value and curriculum is complete
        device: Device to use for training (default: CUDA if available, else CPU)
        clip_grad: Whether to clip gradients during training (default: True)
        use_amp: Whether to use automatic mixed precision (default: True for CUDA)
        l1_lambda: L1 regularization strength (default: 1e-5)
        weight_decay: L2 regularization strength (default: 1e-5)
        is_finetuning: Whether this is a fine-tuning run (default: False)
        momentum: Momentum parameter for SGD when fine-tuning (default: 0.9)
        **kwargs: Additional arguments to pass to the optimizer
    
    Raises:
        TypeError: If dataset is not an instance of ReachabilityDataset
    
    Notes:
        - Training uses different optimization strategies based on is_finetuning:
          - Regular training: Adam optimizer
          - Fine-tuning: SGD with momentum and learning rate scheduling
        - Uses curriculum learning with two phases:
          1. Pretraining phase: Trains on a subset of time values
          2. Curriculum phase: Gradually increases the time range
        - Stopping criteria:
          - Early stopping during fine-tuning if loss plateaus
          - Stopping if loss falls below epsilon and curriculum is complete
        - Saves checkpoints periodically and logs metrics
        - Supports automatic mixed precision training on CUDA devices
        - Applies both L1 and L2 regularization to prevent overfitting
    """
    if not isinstance(dataset, ReachabilityDataset):
        raise TypeError(f"Dataset must be an instance of ReachabilityDataset, got {type(dataset)}")
    
    # Enable automatic mixed precision for CUDA devices
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler() if use_amp else None
    
    # Enable CUDA optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Ensure model and data are on the correct device
    model = model.to(device)
    
    # Adjust optimizer settings based on whether we're fine-tuning
    if is_finetuning:
        # Use SGD with momentum for fine-tuning
        optim = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay * 0.1,  # Reduce regularization during fine-tuning
            **kwargs
        )
        # Create a learning rate scheduler for fine-tuning
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=10000, cooldown=2000, factor=0.5, min_lr=1e-6)
        
        # Add custom learning rate logging callback
        def log_lr(optimizer):
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate adjusted to: {current_lr}")
            
        # Store the callback with the scheduler
        scheduler.log_lr = log_lr
    else:
        # Use Adam for initial training
        optim = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        scheduler = None

    # Initialize curriculum
    curriculum = Curriculum(
        dataset=dataset,
        total_steps=curriculum_epochs,
        time_min=time_min,
        time_max=time_max,
        rollout=not is_finetuning  # Disable rollout during fine-tuning
    )

    # Make sure all path operations use Path consistently
    model_dir = Path(model_dir)
    checkpoints_dir = model_dir / 'checkpoints'

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    model.checkpoint_dir = checkpoints_dir  # Set checkpoint directory for model
    patience = 0
    max_lambda = 0.1

    with tqdm(total=max_epochs) as pbar:
        train_losses = []
        stopping_flag = False
        progress_flag = False # Flag to indicate if curriculum should procude forward
        for epoch in range(0, max_epochs): 
            start_time = time.time()
            # Update curriculum scheduler epoch at the start of each epoch
            curriculum.step(progress_flag)
            
            if stopping_flag or epoch % epochs_til_checkpoint == 0:
                # Save periodic checkpoint using model's method
                if stopping_flag:
                    name = 'model_final'
                    np.savetxt(checkpoints_dir / f'train_losses_final.txt',
                            np.array(train_losses))
                else:
                    name='model_current'
                    np.savetxt(checkpoints_dir / f'train_losses_epoch_{epoch:04d}.txt',
                            np.array(train_losses))
                    
                model.save_checkpoint(
                    name=name,
                    optimizer=optim,
                    epoch=epoch
                )
                train_losses = []
                _, t_max = curriculum.get_time_range()
                if validation_fn is not None:
                    validation_fn(model, checkpoints_dir, epoch, t_max=t_max)

                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
                if stopping_flag:
                    logger.info("Training stopped as loss is below epsilon and curriculum is complete")
                    break

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
                mean_train_loss = sum(loss.mean() * loss_weights.get(name, 1.0)
                                for name, loss in losses.items())
                
                # Apply weights to losses and normalize by batch size
                max_train_loss = sum(loss.max() * loss_weights.get(name, 1.0)
                                for name, loss in losses.items())
                
                train_loss = mean_train_loss + max_lambda*max_train_loss
                
                # Calculate total loss and add L1 regularization using PyTorch's built-in function
                if l1_lambda > 0:
                    l1_loss = torch.tensor(0., device=device)
                    for param in model.parameters():
                        l1_loss += F.l1_loss(param, torch.zeros_like(param), reduction='sum')
                    train_loss += l1_lambda * l1_loss

            dichlet_condition_SAT = losses['dirichlet'].max() < epsilon*0.75
            diff_constraint_SAT =  losses['diff_constraint_hom'].max() < epsilon # Would need to be adapted if time hoizon is not 1

            if dichlet_condition_SAT and (curriculum.is_pretraining or diff_constraint_SAT):
                progress_flag = True
                if curriculum.get_progress() == 1.0 and losses['diff_constraint_hom'].max() < epsilon*0.95:
                    patience += 1
                    stopping_flag = patience > 1000  # Stop after minimum of 1000 consistent epochs
                else:
                    patience = 0
            else:
                progress_flag = False

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

            # Report progress
            if (epoch % epochs_til_checkpoint/10 == 0) or stopping_flag:
                tqdm.write(f"Epoch {epoch}, Total Loss: {train_loss:.6f},"
                          f"L1 Reg: {(l1_lambda * l1_loss if l1_lambda > 0 else 0):.6f}, "
                          f"L2 Reg: {(weight_decay * sum((p ** 2).sum() for p in model.parameters())):.6f}, "
                          f"Time: {time.time() - start_time:.3f}s")
                tqdm.write(f"Diff Constraint Mean: {losses['diff_constraint_hom'].mean():.6f}, "
                          f"Diff Constraint Max: {losses['diff_constraint_hom'].max():.6f}, "
                          f"Dirichlet Max: {losses['dirichlet'].max():.6f}")
                curr_progress = curriculum.get_progress()
                t_min, t_max = curriculum.get_time_range()
                phase = "Pretraining" if curriculum.is_pretraining else "Curriculum"
                tqdm.write(f"{phase} - Progress: {curr_progress:.2%}, Time range: [{t_min:.3f}, {t_max:.3f}]")

            pbar.update(1)

            # Learning rate scheduling for fine-tuning
            if is_finetuning and scheduler is not None:
                prev_lr = optim.param_groups[0]['lr']
                scheduler.step(train_loss)
                # Log if learning rate changed
                if prev_lr != optim.param_groups[0]['lr']:
                    scheduler.log_lr(optim)

        # Save final model
        model.save_checkpoint(
            name='model_final',
            optimizer=optim,
            epoch=max_epochs,
            training_completed=True
        )

        # Save final losses
        np.savetxt(checkpoints_dir / 'train_losses_final.txt', 
                   np.array(train_losses))
