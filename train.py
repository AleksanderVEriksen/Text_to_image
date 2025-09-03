from diffusers import DDPMScheduler
from utils import plot_images, load_to_tensor, load_batch_to_tensor

from torch.utils.data import DataLoader
import torchvision

import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F

from model import UNET, BasicUNet
from tqdm import trange
from torch.amp import autocast

if torch.cuda.is_available() == False:
    sys.exit()

if torch.cuda.is_available():
    autocast_device = 'cuda'
    
else:
    autocast_device = 'cpu'

Test = True

num_epochs = 2
batch_size = 128
max_timesteps = 10

# Load dataset from data.py
from data import get_dataset
if Test == False:
    dataset = get_dataset()
    images = load_batch_to_tensor(dataset, batch_size=batch_size)
else:
    # Load example dataset for testing
    dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Create the UNET model
model = BasicUNet().to(autocast_device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss() 


# Configurate the noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=max_timesteps,
    beta_start=0.0001,
    beta_end=0.02,
)


for epoch in trange(num_epochs):
    for x, y in train_dataloader:
        x = x.to(autocast_device)
        timestep = torch.randint(0, max_timesteps, (x.shape[0],)).to(autocast_device) # Pick random noise step
        noise = torch.rand_like(x).to(autocast_device)  # Generate random noise
        noisy_images = noise_scheduler.add_noise(x, noise, timestep).to(autocast_device)
        with autocast(device_type=autocast_device):
            optimizer.zero_grad()
            noise_pred = model(noisy_images, timestep)
            loss = loss_fn(noise_pred, noise)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Fetch some data
x, y = next(iter(train_dataloader))
x = x[:8].to(autocast_device) # Only using the first 8 for easy plotting

# Corrupt with a range of amounts
amount = torch.linspace(0, 1, x.shape[0]) # Left to right -> more corruption
for i, a in enumerate(amount):
    t = int(a * (max_timesteps - 1))  # Scale to max_timesteps
    timestep = torch.tensor([t])
    noise = torch.randn_like(x)
    noised_x = noise_scheduler.add_noise(x[i:i+1], noise, timestep ) if i == 0 else torch.cat( (noised_x, noise_scheduler.add_noise(x[i:i+1], noise, torch.tensor([t]))), dim=0)
timesteps = torch.tensor([int(a * (max_timesteps - 1)) for a in amount]).to(autocast_device)

# Get the model predictions
with torch.no_grad():
    preds = model(noised_x, timesteps)

import matplotlib.pyplot as plt
# Plot
fig, axs = plt.subplots(3, 1, figsize=(12, 7))
axs[0].set_title('Input data')
axs[0].imshow(torchvision.utils.make_grid(x)[0].cpu().clip(0, 1), cmap='Greys')
axs[1].set_title('Corrupted data')
axs[1].imshow(torchvision.utils.make_grid(noised_x)[0].cpu().clip(0, 1), cmap='Greys')
axs[2].set_title('Network Predictions')
axs[2].imshow(torchvision.utils.make_grid(preds)[0].cpu().clip(0, 1), cmap='Greys')
plt.show()