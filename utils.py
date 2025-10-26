import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

import matplotlib.pyplot as plt
import torchvision
import torch
import numpy as np

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
def sample_images(model, betas, img_size, device, n=16):

    T = len(betas)
    alphas = 1 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)

    x = torch.randn((n, 3, img_size, img_size), device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((n,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            predicted_noise = model(x, t_batch)

        coef1 = sqrt_recip_alphas[t]
        coef2 = sqrt_one_minus_alpha_cumprod[t]

        x0_pred = (x - coef2 * predicted_noise) / coef1

        if t > 0:
            noise = torch.randn_like(x)
            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_cumprod_t = alpha_cumprod[t]
            sigma_t = torch.sqrt(beta_t * (1 - alpha_cumprod_t) / (1 - alpha_t))
            x = torch.sqrt(alpha_t) * x0_pred + sigma_t * noise
        else:
            x = x0_pred
        
        return x  # Returner de genererte bildene