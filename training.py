from tqdm.auto import tqdm
import torch
from torch.amp import autocast, GradScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

def train_epoch(model, dataloader, noise_scheduler, optimizer, loss_fn,
                device, ema=None, grad_clip=1.0, mixed_precision=True):
    """
    One training epoch with tqdm progress bar.
    Predicts added noise (standard DDPM objective).
    """
    model.train()
    scaler = GradScaler(enabled=mixed_precision)
    total_loss = 0.0

    # Use scheduler.config.num_train_timesteps (avoid deprecation)
    max_steps = noise_scheduler.config.num_train_timesteps

    pbar = tqdm(dataloader,
                unit="batch",
                desc="Train",
                leave=False,
                position=0,
                dynamic_ncols=True)
    for batch in pbar:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        timesteps = torch.randint(0, max_steps, (images.size(0),), device=device).long()
        noise = torch.randn_like(images)
        noisy = noise_scheduler.add_noise(images, noise, timesteps)

        with autocast(device_type=device.type, enabled=mixed_precision):
            pred = model(noisy, timesteps, labels=labels)
            loss = loss_fn(pred, noise)  # predict the true noise

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if ema is not None:
            ema.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(1, len(dataloader))