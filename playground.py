from diffusers import DDPMScheduler
from utils import plot_images, load_to_tensor, load_batch_to_tensor

import numpy as np
import torch

# Load dataset from data.py
from data import get_dataset
dataset = get_dataset()

# Configure the noise scheduler
max_timesteps = 10

noise_scheduler = DDPMScheduler(
    num_train_timesteps=max_timesteps,
    beta_start=0.0001,
    beta_end=0.02,
)

# Load a single image from the dataset and convert to tensor
image = load_to_tensor(dataset)

# Generate noisy images
timesteps = torch.arange(0, max_timesteps)
noise = torch.randn(image.shape)
noisy_images = noise_scheduler.add_noise(image, noise, timesteps).numpy()
# Visualize the original and noisy images
plot_images(image, noisy_images, max_images=1)  # Adjusted to show all timesteps

# Load a batch of images from the dataset and convert to tensor
batch_size = 128
images = load_batch_to_tensor(dataset, batch_size=batch_size)
print(f"Batch shape Image: {images.shape}")  # Should be (batch_size, 3, 512, 512)

timesteps = torch.randint(0, max_timesteps, (batch_size,))  # Random timesteps for each image

# Generate noisy images for the batch
noise = torch.randn(images.shape)
noisy_images = []
for i in range(len(images)):
    img = images[i].unsqueeze(0)
    noisy_seq = []
    for t in range(max_timesteps):
        noise = torch.randn(img.shape)
        timestep = torch.tensor([t])
        noisy_image = noise_scheduler.add_noise(img, noise, timestep).numpy()
        noisy_seq.append(noisy_image[0])
    noisy_images.append(noisy_seq)
noisy_images = np.stack(noisy_images, axis=0)  # (batch_size, max_timesteps, 3, 512, 512)
noisy_images = torch.tensor(noisy_images, dtype=torch.float32)

from matplotlib import pyplot as plt
img = np.transpose(noisy_image[0], (1, 2, 0))  # Convert to HWC for plotting
plt.imshow(img)
plt.axis('off')
plt.show()
# Visualize the original and noisy images for the batch
plot_images(images, noisy_images, max_images=1, max_noise=10, steps=1)  # Adjusted to show all timesteps
# Define and test the U-Net model
print(f"Batch shape noisy: {noisy_images.shape}") # (batch_size, max_timesteps, 3, 512, 512)

