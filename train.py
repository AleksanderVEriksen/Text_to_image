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

num_epochs = 20
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
model = BasicUNet(in_channels=1, out_channels=1).to(autocast_device)

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

# Save the model after training
torch.save(model.state_dict(), 'model.pth')
