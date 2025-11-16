import matplotlib.pyplot as plt
from diffusers import DDPMScheduler
import torch
import torchvision
from torchvision import transforms
import PIL.Image
from tqdm.auto import tqdm
import numpy as np
import os
import torch.nn.functional as F
from torchvision.models import inception_v3


# Definer transformasjoner én gang
transform = transforms.Compose([
    transforms.ToTensor(),           # convert to tensor and normalize to [0,1]
    transforms.Lambda(lambda x: 2 * x - 1)  # normalize to [-1, 1]
])

# Load image to tensor
def load_single_img_to_tensor(dataset):
    sample = next(iter(dataset))
    image = sample['jpg']
    image = transform(image)  # (C, H, W), normalisert til [0, 1]
    return image

# Load dataset of images to tensor
def sample_to_tensor(sample):
    if isinstance(sample, dict):
        return transform(sample['jpg'])
    else:
        return transform(sample[0])
    
def collate_fn(batch):
    images = []
    labels = []
    for sample in batch:
        if isinstance(sample, dict):
            images.append(sample.get('image', sample.get('jpg', None)))
            labels.append(sample.get('label', -1))
        else:
            # Handle tuple case (image, label)
            if isinstance(sample, tuple) and len(sample) >= 1:
                img = sample[0]
                label = sample[1] if len(sample) > 1 else -1
                if isinstance(img, PIL.Image.Image):
                    img = transform(img)
                images.append(img)
                labels.append(label)
            else:
                images.append(sample)
                labels.append(-1)

    # Stack images and ensure they're tensors
    images = [img if isinstance(img, torch.Tensor) else transform(img) for img in images]
    images = torch.stack(images, dim=0)
    
    # Convert labels to tensor
    labels = torch.tensor([int(l) if not isinstance(l, torch.Tensor) else int(l.item()) 
                        for l in labels], dtype=torch.long)
    return images, labels


# Sample generated images
# use scheduler.set_timesteps(...) before calling this
def text_to_label(label, max_num_classes: int = 10):
    """Convert text/numeric label to tensor index."""
    if isinstance(label, int):
        return label
    if isinstance(label, str):
        try:
            # Try direct numeric conversion first
            return int(label)
        except ValueError:
            # Map text to numbers
            if max_num_classes > 10:
                raise ValueError("Text labels not supported for num_classes > 10")
            label_map = {
                'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
                'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
            }
            label = label.lower()
            if label in label_map:
                return label_map[label]
    raise ValueError(f"Unsupported label format: {label}")

def sample_images(model, scheduler, img_size, device, n=16, Test=False, debug=False, labels=None, num_classes=10):
    """Generate images using the diffusion model."""
    model.eval()
    
    # Start with black noise (negative values) instead of random noise
    #x = -torch.abs(torch.randn((n, 1 if Test else 3, img_size, img_size), device=device))
    
    # Start with random noise (standard Gaussian)
    x = torch.randn((n, 1 if Test else 3, img_size, img_size), device=device)

    if debug:
        print("\nSampling Progress:")
        print("-" * 50)
    timesteps_used = list(scheduler.timesteps)
    # Sampling loop
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = model(x, t.expand(n).to(device), labels)
            step_output = scheduler.step(noise_pred, t, x)
            x = step_output.prev_sample
            
            
            if debug and t % 100 == 0:
                print(f"Step {t:3d}/{scheduler.timesteps[0]}: min={x.min():6.3f}, max={x.max():6.3f}")
    
    if debug:
        print("-" * 50)
        print(f"Final range: [{x.min():6.3f}, {x.max():6.3f}]")
    
    # Ensure final output has proper contrast
    samples_cpu = x.cpu()
    timesteps_tensor = torch.tensor(timesteps_used, dtype=torch.long, device=device).cpu()
    return samples_cpu, timesteps_tensor


def validate(model, epochs, val_dataloader, noise_scheduler, loss_fn, device, max_batches=None, calculate_fid_score=False, fid_epoch_calc=10,img_size=28, Test=True):
    """Run validation loop and return loss metrics, optionally including FID score."""
    model.eval()
    val_loss = 0
    running_avg_loss = 0
    alpha = 0.1  # Smoothing factor for running average
    batches = 0
    fid_scores = [] if calculate_fid_score else None
    
    with torch.no_grad():
        for batch in tqdm(
                val_dataloader, 
                desc="Validating", 
                leave=False, 
                dynamic_ncols=True, 
                position=0
                ):
            if max_batches and batches >= max_batches:
                break
                
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            # Calculate validation loss
            timesteps = torch.randint(0, len(noise_scheduler.timesteps), 
                                    (images.shape[0],), device=device).long()
            
            noise = torch.randn_like(images)
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            
            noise_pred = model(noisy_images, timesteps, labels=labels)
            loss = loss_fn(noise_pred, noise)
            
            val_loss += loss.item()
            # Update running average
            if batches == 0:
                running_avg_loss = loss.item()
            else:
                running_avg_loss = (1 - alpha) * running_avg_loss + alpha * loss.item()
            batches += 1
            
            # Optionally calculate FID for this batch every 50 epochs
            if epochs % fid_epoch_calc == 0 and calculate_fid_score:
                if batches <= max_batches:
                    # Generate samples matching the batch
                    generated_samples, timesteps_tensor = sample_images(
                        model, noise_scheduler, img_size, device, 
                        n=images.shape[0], Test=Test, debug=False, 
                        labels=labels, num_classes=10
                    )
                    generated_samples = generated_samples.to(device)
                    # Calculate FID for this batch
                    fid = calculate_fid(images, generated_samples)
                    fid_scores.append(fid)
    
    results = {
        'val_loss': val_loss / batches,
        'running_avg_loss': running_avg_loss
    }
    
    if calculate_fid_score:
        results['fid_score'] = np.mean(fid_scores) if fid_scores else float('inf')
    
    return results

def tensor_grid_to_numpy(tensor, nrow=8, rescale=True):
    """Make a torchvision grid and return a float32 numpy array.
    If rescale=True normalize to [0,1] using min/max. If rescale=False clip to [0,1]."""
    # Accept either torch.Tensor or numpy array
    if isinstance(tensor, np.ndarray):
        arr = tensor
        # If HWC -> CHW expectation already handled outside; ensure float32
        arr = arr.astype(np.float32)
        # If array is HWC single-channel -> squeeze last dim
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        # Rescale or clip
        mn, mx = float(arr.min()), float(arr.max())
        if rescale:
            if mn < 0.0 or mx > 1.0:
                if mx - mn > 1e-8:
                    arr = (arr - mn) / (mx - mn)
                else:
                    arr = np.clip(arr, 0.0, 1.0)
        else:
            arr = np.clip(arr, 0.0, 1.0)
        return arr
    # torch.Tensor path
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=False, pad_value=1)
    # move to CPU, ensure float32
    grid = grid.detach().cpu().to(torch.float32)
    arr = grid.permute(1, 2, 0).numpy().astype(np.float32)
    # if single-channel, squeeze last dim
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    # Rescale to [0,1] if necessary
    mn, mx = float(arr.min()), float(arr.max())
    if rescale:
        if mn < 0.0 or mx > 1.0:
            if mx - mn > 1e-8:
                arr = (arr - mn) / (mx - mn)
            else:
                arr = np.clip(arr, 0.0, 1.0)
    else:
        arr = np.clip(arr, 0.0, 1.0)
    return arr

def normalize_per_sample(tensor, min_val=-1, max_val=1):
    """Normalize each sample in a batch independently to a given range.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (N,C,H,W)
        min_val (float): Target minimum value
        max_val (float): Target maximum value
    
    Returns:
        torch.Tensor: Normalized tensor
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")
        
    # Get dimensions
    B, C, H, W = tensor.shape
    
    # Reshape to (B,C*H*W) for per-sample normalization
    flat = tensor.view(B, -1)
    
    # Get min/max per sample
    min_per_sample = flat.min(dim=1, keepdim=True)[0]
    max_per_sample = flat.max(dim=1, keepdim=True)[0]
    
    # Normalize
    scale = (max_val - min_val) / (max_per_sample - min_per_sample + 1e-8)
    normalized = (flat - min_per_sample) * scale + min_val
    
    # Reshape back
    return normalized.view(B, C, H, W)

def load_model_weights(model, batch_size, model_name, device, is_test=False, is_checkpoint=False, is_ema=False):
    """Load model weights with proper error handling and path resolution.
    
    Args:
        model: The PyTorch model to load weights into
        model_name (str): Name of the model/weights file
        device: PyTorch device to load weights to
        is_test (bool): If True, look in test model directory
        is_checkpoint (bool): If True, load from checkpoints folder
        is_ema (bool): If True, load EMA weights
    
    Returns:
        dict or None: Full checkpoint dict if available, else None
    """
    # Determine weight path
    if is_checkpoint:
        path = os.path.join(f"models/{batch_size}", "checkpoints", f"{model_name}.pth")
    elif is_ema:
        path = os.path.join(f"models/{batch_size}", "EMA", f"{model_name}_EMA.pth")
    else:
        path = os.path.join(f"models/{batch_size}", f"{model_name}.pth")
    
    try:
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model state from checkpoint: {path}")
            return checkpoint
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights from: {path}")
            return None
            
    except FileNotFoundError:
        print(f"No weights found at {path}")
        return None
    except Exception as e:
        print(f"Error loading weights from {path}: {str(e)}")
        return None

def plot_losses(train_losses, val_losses, val_every, save_path="figures/loss_curves/loss_plot.png"):
    """Plot training and validation losses over epochs.
    
    Args:
        train_losses (list): List of training losses for each epoch
        val_losses (list): List of validation losses (every val_every epochs)
        val_every (int): Frequency of validation (e.g., every 5 epochs)
        save_path (str): Path to save the plot image
    """
    epochs = list(range(1, len(train_losses) + 1))
    val_epochs = [i * val_every for i in range(1, len(val_losses) + 1)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_epochs, val_losses, label='Validation Loss', color='red', linewidth=2, marker='o')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {save_path}")

def calculate_fid(real_images, generated_images, device='cuda'):
    """Calculate Fréchet Inception Distance between real and generated images.
    
    Args:
        real_images (torch.Tensor): Real images, shape (N, C, H, W)
        generated_images (torch.Tensor): Generated images, shape (N, C, H, W)
        device (str): Device to run calculations on
    
    Returns:
        float: FID score
    """
    # For MNIST, use a simpler metric since InceptionV3 isn't ideal for grayscale
    # Calculate MSE as a proxy for now (replace with proper FID later)
    mse = F.mse_loss(real_images, generated_images).item()
    
    # Placeholder for proper FID implementation:
    # - Extract features using InceptionV3
    # - Calculate mean and covariance of features
    # - Compute Fréchet distance
    
    return mse  # Return MSE for now, implement full FID when needed

def save_with_retry(save_func, *args, **kwargs):
    """Helper function to save files with automatic directory creation on failure."""
    try:
        save_func(*args, **kwargs)
    except (OSError, FileNotFoundError, RuntimeError) as e:  # Add RuntimeError
        # Extract the file path from args or kwargs
        file_path = None
        if args and isinstance(args[1], str):  # Common pattern: save_func(tensor, path, ...)
            file_path = args[1]
        elif 'path' in kwargs:
            file_path = kwargs['path']
        elif 'save_path' in kwargs:
            file_path = kwargs['save_path']
        
        if file_path:
            dir_path = os.path.dirname(file_path)
            print(f"Directory not found for {file_path}, creating it: {e}")
            os.makedirs(dir_path, exist_ok=True)
            save_func(*args, **kwargs)  # Retry after creating directory
        else:
            raise e  # Re-raise if we can't determine the path
        
def timesteps_to_str(ts, max_items=20):
    lst = list(ts.tolist())
    if len(lst) > max_items:
        return ", ".join(map(str, lst[:max_items])) + ", ..."
    return ", ".join(map(str, lst))