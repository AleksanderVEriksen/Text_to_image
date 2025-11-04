from tqdm.auto import tqdm
import torch
from torch.cuda.amp import autocast


def train_epoch(model, dataloader, optimizer, scheduler, noise_scheduler, loss_fn, 
                device, scaler, epoch, num_epochs):
    """Single epoch training function"""
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(enumerate(dataloader), 
                    desc=f"Epoch {epoch+1}/{num_epochs}",
                    leave=False,
                    position=0,
                    dynamic_ncols=True,
                    total=len(dataloader))
    
    for step, (images, labels) in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        timesteps = torch.randint(0, len(noise_scheduler.timesteps), 
                                (images.shape[0],), device=device).long()
        
        noise = torch.randn_like(images)
        noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            noise_pred = model(noisy_images, timesteps, labels=labels)
            loss = loss_fn(noise_pred, noise)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{epoch_loss/(step+1):.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
        }, refresh=True)
    
    progress_bar.close()
    return epoch_loss / len(dataloader)