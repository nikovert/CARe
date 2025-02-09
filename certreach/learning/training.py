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

logger = logging.getLogger(__name__)

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          epochs: int, 
          lr: float, 
          steps_til_summary: int, 
          epochs_til_checkpoint: int, 
          model_dir: str, 
          loss_fn: Callable, 
          summary_fn: Optional[Callable] = None, 
          val_dataloader: Optional[torch.utils.data.DataLoader] = None, 
          double_precision: bool = False, 
          clip_grad: bool = False, 
          use_lbfgs: bool = False, 
          loss_schedules: Optional[Dict[str, Callable]] = None, 
          validation_fn: Optional[Callable] = None, 
          start_epoch: int = 0,
          device: Optional[torch.device] = None,
          use_amp: bool = True) -> None:

    # Use provided device or default to CUDA if available
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize gradient scaler for AMP without device_type
    scaler = torch.GradScaler() if use_amp and device.type == 'cuda' else None

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    if use_lbfgs:
        optim = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')

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
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(start_epoch, epochs):
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

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                # Move data to device after loading, handling CPU pinned memory correctly
                model_input = {key: value.to(device, non_blocking=True) for key, value in model_input.items()}
                gt = {key: value.to(device, non_blocking=True) for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                if use_lbfgs:
                    def closure():
                        optim.zero_grad()
                        model_input['coords'].requires_grad_(True)  # Ensure gradients
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt)
                        train_loss = sum(loss.mean() for loss in losses.values())
                        train_loss.backward()
                        return train_loss
                    optim.step(closure)
                else:
                    optim.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                    
                    # Updated autocast context manager
                    with torch.autocast(device.type, enabled=use_amp):
                        model_input['coords'].requires_grad_(True)  # Ensure gradients
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt)

                        train_loss = sum(loss.mean() for loss in losses.values())

                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    if loss_schedules and loss_name in loss_schedules:
                        writer.add_scalar(f"{loss_name}_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)
                    writer.add_scalar(loss_name, single_loss, total_steps)

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                if total_steps % steps_til_summary == 0:
                    # Save current state
                    model.save_checkpoint(
                        name='model_current',
                        optimizer=optim,
                        epoch=epoch,
                        step=total_steps
                    )

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

                pbar.update(1)

                if total_steps % steps_til_summary == 0:
                    tqdm.write(f"Epoch {epoch}, Total loss {train_loss:.6f}, iteration time {time.time() - start_time:.6f}")

                    if val_dataloader:
                        logger.info("Running validation set...")
                        model.eval()
                        val_losses = []
                        # Updated autocast for validation
                        with torch.no_grad(), torch.autocast(device.type, enabled=use_amp):
                            for val_input, val_gt in val_dataloader:
                                val_input = {k: v.to(device, non_blocking=True) for k, v in val_input.items()}
                                val_gt = {k: v.to(device, non_blocking=True) for k, v in val_gt.items()}
                                val_output = model(val_input)
                                val_batch_losses = loss_fn(val_output, val_gt)
                                val_losses.append(sum(loss.mean() for loss in val_batch_losses.values()))
                            
                            avg_val_loss = torch.stack(val_losses).mean()
                            writer.add_scalar("val_loss", avg_val_loss.item(), total_steps)
                        model.train()

                total_steps += 1

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


class LinearDecaySchedule:
    def __init__(self, start_val: float, final_val: float, num_steps: int):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, step: int) -> float:
        return self.start_val + (self.final_val - self.start_val) * min(step / self.num_steps, 1.0)
