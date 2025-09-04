from diffusers import DDPMScheduler
from utils import plot_images, load_to_tensor, load_batch_to_tensor

import numpy as np
import torch
from matplotlib import pyplot as plt

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
images = load_batch_to_tensor(dataset, batch_size=16)  # Get the first image
print(f"Single image shape: {images[0].shape}")  # Should be (3, 128, 128)
plt.imshow(images[0].permute(1,2,0).numpy())
plt.title("Original Image")
plt.axis('off')
plt.show()
# Generate noisy images
timestep = torch.randint(0, max_timesteps, (images.shape[0],)) # Pick random noise step
noise = torch.randn(images.shape)
noisy_images = noise_scheduler.add_noise(images, noise, timestep)
# Visualize the original and noisy images
plot_images(images, noisy_images, max_images=2)  # Adjusted to show all timesteps

# Load a batch of images from the dataset and convert to tensor
batch_size = 16
images = load_batch_to_tensor(dataset, batch_size=batch_size)
print(f"Batch shape Image: {images.shape}")  # Should be (batch_size, 3, 512, 512)

timesteps = torch.randint(0, max_timesteps, (batch_size,))  # Random timesteps for each image

# Generate noisy images for the batch
noise = torch.randn(images.shape)
noisy_image = noise_scheduler.add_noise(images, noise, timesteps)

from matplotlib import pyplot as plt
img = np.transpose(noisy_image[0], (1, 2, 0))  # Convert to HWC for plotting
plt.imshow(img)
plt.axis('off')
plt.show()
# Visualize the original and noisy images for the batch
plot_images(images, noisy_images, max_images=1, max_noise=10, steps=1)  # Adjusted to show all timesteps
# Define and test the U-Net model
print(f"Batch shape noisy: {noisy_images.shape}") # (batch_size, max_timesteps, 3, 512, 512)

