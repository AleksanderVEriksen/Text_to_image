from diffusers import DDPMScheduler
from utils import load_batch_to_tensor

from torch.utils.data import DataLoader
import torchvision

import torch
import sys
import torch.nn as nn
import torch, gc

from model import UNET, BasicUNet
from tqdm import trange

# Automatic Mixed Precision - saves memory and speeds up training
from torch.amp import autocast, GradScaler
import argparse
from data import get_dataset

# Clear cache
gc.collect()
torch.cuda.empty_cache()
# -----------------------------------------------
if torch.cuda.is_available() == False:
    sys.exit()

if torch.cuda.is_available():
    autocast_device = 'cuda'
    
else:
    autocast_device = 'cpu'
# ----------------------------------------------

# Sets scaler
scaler = GradScaler()
# ----------------------------------------------

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train UNet on MNIST or custom dataset")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--max_timesteps", type=int, default=10, help="Number of timesteps")
parser.add_argument("--test", action="store_true", help="Use MNIST test dataset")
parser.add_argument("--model", type=str, default="UNET", help="Model type: UNET or Basic")
args = parser.parse_args()
# ----------------------------------------------

if __name__ == "__main__":

    batch_size = args.batch_size
    num_epochs = args.epochs
    Test = args.test

    # Load dataset from data.py
    if args.test == False:
        print("Training on custom dataset")
        dataset = get_dataset()
        dataset = load_batch_to_tensor(dataset, batch_size)
        train_dataloader = DataLoader(dataset, batch_size)
        channels = 3
    else:
        # Load example dataset for testing
        print("Testing on MNIST dataset")
        dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True)
        dataset = load_batch_to_tensor(dataset, batch_size)
        train_dataloader = DataLoader(dataset, batch_size, shuffle=True)
        channels = 1
    # Create the UNET model
    if args.model == "Basic":
        model = BasicUNet(in_channels=1, out_channels=1).to(autocast_device)
    else:
        model = UNET(channels, channels).to(autocast_device)
    print("Input channels: ", channels)

    batch_size = args.batch_size
    num_epochs = args.epochs
    max_timesteps = args.max_timesteps
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss() 

    # Configurate the noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=max_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
    )

    # Training loop
    for epoch in trange(num_epochs):
        model.train()
        for x in train_dataloader:
            x = x.to(autocast_device) # (16, 3, 128, 128) or (8, 1, 28, 28)
            timestep = torch.randint(0, max_timesteps, (x.shape[0],)).to(autocast_device) # Pick random noise step
            noise = torch.rand_like(x).to(autocast_device)  # Generate random noise
            noisy_images = noise_scheduler.add_noise(x, noise, timestep).to(autocast_device)
            # Predict the noise using the model - the model learns to denoise
            # Sets autocasting for the forward pass (model + loss)
            # Sets scales for the backward pass
            with autocast(device_type=autocast_device):
                optimizer.zero_grad()
                noise_pred = model(noisy_images, timestep) # (16, 3, 128, 128) or (8, 1, 28, 28)
                loss = loss_fn(noise_pred, noise)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Save the model after training
    if Test: 
        torch.save(model.state_dict(), './models/model_test.pth') 
    else: 
        torch.save(model.state_dict(), './models/model.pth')
        