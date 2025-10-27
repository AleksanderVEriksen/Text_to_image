import matplotlib.pyplot as plt
from diffusers import DDPMScheduler
import torch
import torchvision

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


from torchvision import transforms

# Definer transformasjoner én gang
transform = transforms.Compose([
    transforms.Resize((128, 128)),   # resize til 128x128
    transforms.ToTensor(),           # konverter til tensor og normaliser til [0,1]
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
    images = [sample_to_tensor(img) for img in batch]
    return torch.stack(images, dim=0)


# Sample generated images
# use scheduler.set_timesteps(...) before calling this
def sample_images(model, scheduler: DDPMScheduler, img_size: int, device, n: int = 16, Test: bool = False, debug: bool = False):
    """
    Use diffusers scheduler for sampling. Call scheduler.set_timesteps(num_inference_steps)
    before calling this function. Returns CPU tensor (n,C,H,W).
    """
    model.eval()
    channels = 1 if Test else 3
    x = torch.randn((n, channels, img_size, img_size), device=device)
    # scheduler.timesteps must be set by caller (scheduler.set_timesteps)
    noise_scheduler = scheduler(num_train_timesteps=scheduler.config.num_train_timesteps, beta_start=scheduler.config.beta_start, beta_end=scheduler.config.beta_end)
    noise_scheduler.set_timesteps(scheduler.config.num_train_timesteps)
    timesteps = list(scheduler.timesteps)   # descending
    for i, t in enumerate(timesteps):
        t_int = int(t)
        t_batch = torch.full((n,), t_int, device=device, dtype=torch.long)
        with torch.no_grad():
            eps_pred = model(x, t_batch)
        out = scheduler.step(model_output=eps_pred, timestep=t_int, sample=x)
        x = out.prev_sample if hasattr(out, "prev_sample") else out["prev_sample"]

        if debug:
            print(f"step {i}/{len(timesteps)-1} t={t_int} | x stats min/max/mean/std:",
                float(x.min()), float(x.max()), float(x.mean()), float(x.std()))
    return x.cpu()
