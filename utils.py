import matplotlib.pyplot as plt
from diffusers import DDPMScheduler
import torch
import torchvision
from torchvision import transforms
import PIL.Image
from tqdm.auto import tqdm
import numpy as np
import os

# ! Not used - possible removal in future
def plot_images(normal_images, noisy_images, max_images=1, max_noise=5, steps=1):
    """
    normal_images: tensor [B,C,H,W] eller [C,H,W]
    noisy_images: tensor [B,T,C,H,W] eller [T,C,H,W]
    max_images: maks antall bilder fra batch å vise
    max_noise: maks antall støy-step å vise per bilde
    steps: steg mellom støy-visning
    """

    # Single image -> batch
    if normal_images.ndim == 3:
        normal_images = normal_images.unsqueeze(0)  # [1,C,H,W]
    if noisy_images.ndim == 4:  # [T,C,H,W]
        noisy_images = noisy_images.unsqueeze(0)  # [1,T,C,H,W]

    B = min(max_images, normal_images.shape[0])

    for b in range(B):
        orig = normal_images[b]          # [C,H,W]
        noisy = noisy_images[b]          # [T,C,H,W]

        T = min(noisy.shape[0], max_noise)
        indices = list(range(0, T, steps))
        num_rows = 1 + len(indices)      # 1 rad for original + 1 rad per step

        fig, axes = plt.subplots(num_rows, 1, figsize=(5, 5*num_rows))
        if num_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Originalbilde øverst
        img = orig.cpu()
        grid = torchvision.utils.make_grid(img, nrow=4, normalize=True)
        axes[0].imshow(grid.permute(1,2,0).numpy())
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Noisy steg under originalen
        for idx, i in enumerate(indices):
            img = noisy[i]
            if img.ndim == 2:
                img = img.unsqueeze(0)
            img = img.cpu()
            grid = torchvision.utils.make_grid(img, nrow=4, normalize=True)
            axes[idx+1].imshow(grid.permute(1,2,0).numpy())
            axes[idx+1].set_title(f"Step {i+1}")
            axes[idx+1].axis('off')

        plt.tight_layout()
        plt.show()


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
def text_to_label(label, num_classes: int = 10):
    """Convert text/numeric label to tensor index."""
    if isinstance(label, int):
        return label
    if isinstance(label, str):
        try:
            # Try direct numeric conversion first
            return int(label)
        except ValueError:
            # Map text to numbers
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
    x = -torch.abs(torch.randn((n, 1 if Test else 3, img_size, img_size), device=device))
    
    if debug:
        print("\nSampling Progress:")
        print("-" * 50)
    
    # Sampling loop
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = model(x, t.expand(n).to(device), labels)
            step_output = scheduler.step(noise_pred, t, x)
            x = step_output.prev_sample
            
            # Ensure values stay mostly negative for black appearance
            x = x - x.mean(dim=(2,3), keepdim=True)
            
            if debug and t % 20 == 0:
                print(f"Step {t:3d}/{scheduler.timesteps[0]}: min={x.min():6.3f}, max={x.max():6.3f}")
    
    if debug:
        print("-" * 50)
        print(f"Final range: [{x.min():6.3f}, {x.max():6.3f}]")
    
    # Ensure final output has proper contrast
    x = torch.tanh(x)  # Squash to [-1, 1]
    return x.cpu()


def validate(model, val_dataloader, noise_scheduler, loss_fn, device, max_batches=None):
    """Run validation loop and return both batch and running average losses."""
    model.eval()
    val_loss = 0
    running_avg_loss = 0
    alpha = 0.1  # Smoothing factor for running average
    batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating", leave=False):
            if max_batches and batches >= max_batches:
                break
                
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
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
            
    return val_loss / batches, running_avg_loss

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

def load_model_weights(model, model_name, device, is_test=False, is_checkpoint=False, is_ema=False):
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
        path = os.path.join("models", "checkpoints", f"{model_name}.pth")
    elif is_ema:
        path = os.path.join("models", "EMA", f"{model_name}_EMA.pth")
    else:
        path = os.path.join("models", f"{model_name}.pth")
    
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