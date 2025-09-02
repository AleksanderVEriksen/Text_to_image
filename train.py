from diffusers import DDPMScheduler

from utils import plot_images, load_to_tensor, load_batch_to_tensor

import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F

from model import UNET
from tqdm import trange
from model import UNET
from torch.cuda.amp import autocast

if torch.cuda.is_available() == False:
    sys.exit()

if torch.cuda.is_available():
    autocast_device = 'cuda'
    
else:
    autocast_device = 'cpu'

# Load dataset from data.py
from data import get_dataset
dataset = get_dataset()

# Create the UNET model
model = UNET().to(autocast_device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss() 

num_epochs = 10
batch_size = 128
max_timesteps = 10
# Configurate the noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=max_timesteps,
    beta_start=0.0001,
    beta_end=0.02,
)

for epoch in trange(num_epochs):
    images = load_batch_to_tensor(dataset, batch_size=batch_size)
    timesteps = torch.randint(0, max_timesteps, (batch_size,))
    noise = torch.rand_like(images)
    noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
    with autocast(device_type=autocast_device):
        optimizer.zero_grad()
        noise_pred = model(noisy_images, timesteps)
        loss = loss_fn(noise_pred, noise)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")