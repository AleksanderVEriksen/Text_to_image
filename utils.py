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
def sample_images(model, scheduler: DDPMScheduler, img_size: int, device, n: int = 16, Test: bool = False, debug: bool = False, save_intermediates: bool = False):
    """
    Use diffusers scheduler for sampling. Call scheduler.set_timesteps(num_inference_steps)
    before calling this function. Returns CPU tensor (n,C,H,W).

    debug: print per-step stats (already present).
    save_intermediates: save x0_pred grids at a few checkpoints (helps inspect progressive denoising).
    """
    model.eval()
    channels = 1 if Test else 3
    x = torch.randn((n, channels, img_size, img_size), device=device)

    # prepare beta / alpha arrays from scheduler (device-aware)
    betas = scheduler.betas.to(device) if hasattr(scheduler, "betas") else torch.linspace(scheduler.beta_start, scheduler.beta_end, scheduler.config.num_train_timesteps, device=device)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)

    timesteps = list(scheduler.timesteps)
    saved = {}
    for i, t in enumerate(timesteps):
        t_int = int(t)
        t_batch = torch.full((n,), t_int, device=device, dtype=torch.long)
        with torch.no_grad():
            eps_pred = model(x, t_batch)

        # compute x0 estimate for debugging/visualization (per-sample)
        denom = sqrt_alpha_cumprod[t_int].view(1, 1, 1, 1)
        x0_pred = (x - sqrt_one_minus_alpha_cumprod[t_int].view(1,1,1,1) * eps_pred) / (denom + 1e-8)

        out = scheduler.step(model_output=eps_pred, timestep=t_int, sample=x)
        x = out.prev_sample if hasattr(out, "prev_sample") else out["prev_sample"]

        if debug:
            print(f"step {i}/{len(timesteps)-1} t={t_int} | x stats min/max/mean/std: {float(x.min()):.6f}/{float(x.max()):.6f}/{float(x.mean()):.6f}/{float(x.std()):.6f}")

        # optionally save intermediate x0_pred visualizations at a few checkpoints
        if save_intermediates and (i in (0, len(timesteps)//4, len(timesteps)//2, 3*len(timesteps)//4, len(timesteps)-1)):
            # normalize each sample independently for visibility (keeps relative structure)
            xp = x0_pred.detach().cpu().clone()
            N = xp.shape[0]
            for j in range(N):
                s = xp[j]
                mn, mx = float(s.min()), float(s.max())
                if mx - mn > 1e-8:
                    xp[j] = (s - mn) / (mx - mn)
                else:
                    xp[j] = s - mn
            grid = torchvision.utils.make_grid(xp, nrow=int(max(1, min(8, N//2))), normalize=False, pad_value=1)
            saved[f"step_{t_int}"] = grid.permute(1,2,0).numpy()

    # final sample on CPU
    return x.cpu(), saved
