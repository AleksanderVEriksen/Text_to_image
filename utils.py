import matplotlib.pyplot as plt
from diffusers import DDPMScheduler
import torch
import torchvision
from torchvision import transforms
import PIL.Image
from tqdm.auto import tqdm

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
    
    # Set up initial noise
    channels = 1 if Test else 3
    x = torch.randn((n, channels, img_size, img_size), device=device)
    
    # Convert labels to tensor
    if labels is not None:
        if isinstance(labels, (str, int)):
            # Single label -> convert and repeat
            label_idx = text_to_label(labels, num_classes)
            labels = torch.full((n,), label_idx, dtype=torch.long, device=device)
        elif isinstance(labels, (list, tuple)):
            # List of labels -> convert each and create tensor
            label_indices = [text_to_label(l, num_classes) for l in labels]
            labels = torch.tensor(label_indices, dtype=torch.long, device=device)
            if len(labels) < n:
                labels = labels.repeat(n // len(labels) + 1)[:n]
        elif isinstance(labels, torch.Tensor):
            labels = labels.to(device)
        else:
            raise TypeError(f"Unsupported label type: {type(labels)}")
    
    # Sampling loop
    for t in scheduler.timesteps:
        with torch.no_grad():
            # Get model prediction
            noise_pred = model(x, t.expand(n).to(device), labels)
            
            # Update sample with scheduler
            step_output = scheduler.step(noise_pred, t, x)
            x = step_output.prev_sample
            
            if debug:
                print(f"Step {t}: x range [{x.min():.3f}, {x.max():.3f}]")
    
    # Ensure output is in [-1, 1]
    x = torch.clamp(x, -1, 1)
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