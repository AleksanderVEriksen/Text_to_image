import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_images(normal_images, noisy_images, max_images, max_noise=5, steps=1):

    # Handle single image case
    if normal_images.ndim == 3:
        normal_images = normal_images.unsqueeze(0)
    if isinstance(noisy_images, np.ndarray):
        noisy_images = torch.from_numpy(noisy_images)
    if noisy_images.ndim == 4:
        noisy_images = noisy_images.unsqueeze(0)

    # torch.Size([3, 512, 512]) torch.Size([4, 3, 512, 512])
    for x in range(0, max_images):
        image = normal_images[x]
        noisy = noisy_images[x]

        # Remove batch dimension if present
        if image.ndim == 4:
            image = image[0]
        if isinstance(noisy, np.ndarray):
            noisy = torch.from_numpy(noisy)
        if noisy.ndim == 5:
            noisy = noisy[:, 0]  # remove batch dimension

        T = min(noisy.shape[0], max_noise)
        indices = list(range(0, T, steps))
        num_plots = len(indices) + 1  # +1 for the original image
        fig, axes = plt.subplots(1, num_plots, figsize=(3 * (num_plots), 3))

        axes[0].imshow(np.clip(image.permute(1, 2, 0).numpy(), 0, 1))
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        for idx, i in enumerate(indices):
            img = noisy[i]
            if img.ndim == 3:
                img = img.permute(1, 2, 0).numpy()
            elif img.ndim == 2:
                img = img.numpy()  # No channel dimension
            else:
                raise ValueError(f"Unexpected noisy image shape: {img.shape}")
            img = np.clip(img, 0, 1)
            axes[idx + 1].imshow(img)
            axes[idx + 1].set_title(f"Step {i+1}")
            axes[idx + 1].axis('off')
        plt.tight_layout()
        plt.show()

# Load image to tensor
def load_to_tensor(dataset):
    sample = next(iter(dataset))
    image = sample['jpg']
    image = image.resize((512, 512))
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Convert to tensor and add batch dimension
    return image

# Load batch of images to tensor
def load_batch_to_tensor(dataset, batch_size=8):
    images = []
    data_iter = iter(dataset)
    for _ in range(batch_size):
        sample = next(data_iter)
        image = sample['jpg'].resize((512, 512))
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1)  # (C, H, W)
        images.append(image)
    batch = torch.stack(images, dim=0)  # (batch_size, C, H, W)
    return batch