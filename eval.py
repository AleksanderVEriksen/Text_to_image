import torch
import torchvision
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from diffusers import DDPMScheduler
from model import BasicUNet, UNET
from data import get_dataset
from utils import collate_fn
import argparse

if torch.cuda.is_available():
    autocast_device = 'cuda'
    
else:
    autocast_device = 'cpu'



# Parse command line arguments
parser = argparse.ArgumentParser(description="Train UNet on MNIST or custom dataset")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--max_timesteps", type=int, default=10, help="Number of timesteps")
parser.add_argument("--test", action="store_true", help="Use MNIST test dataset")
parser.add_argument("--model", type=str, default="UNET", help="Model type: UNET or Basic")
args = parser.parse_args()
# ----------------------------------------------

batch_size = args.batch_size
max_timesteps = args.max_timesteps
Test = args.test

# Load dataset from data.py
if Test == False:
    print("\n---Training on custom dataset---\n")
    train, val, test = get_dataset()


    train_dataloader = DataLoader(train, batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataloader = DataLoader(val, batch_size, collate_fn=collate_fn, shuffle=True)
    test_dataloader = DataLoader(test, batch_size, collate_fn=collate_fn, shuffle=False)

else:
    # Load example dataset for testing
    print("\n---Testing on MNIST dataset---\n")
    train_ = torchvision.datasets.MNIST(root="mnist/", train=True, download=True)
    test_ = torchvision.datasets.MNIST(root="mnist/", train=False, download=True)

    train_size = int((1 - len(train_)*0.8))
    val_size = (len(train_)-train_size)
    train, val = random_split(train_, [train_size, val_size] )


    train_dataloader = DataLoader(train, batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataloader = DataLoader(val, batch_size, collate_fn=collate_fn, shuffle=True)
    test_dataloader = DataLoader(test_, batch_size, collate_fn=collate_fn, shuffle=False)


# Create the UNET model
if args.model == "Basic": model = BasicUNet(in_channels= 1 if Test else 3, out_channels=1 if Test else 3).to(autocast_device)
else: model = UNET(in_channels= 1 if Test else 3, out_channels=1 if Test else 3).to(autocast_device)

if args.test:
    # Load the trained model weights
    model.load_state_dict(torch.load('models/model_test.pth', weights_only=True))
else:
    # Load the trained model weights
    model.load_state_dict(torch.load('models/model.pth', weights_only=True))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss() 

# Configurate the noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=max_timesteps,
    beta_start=0.0001,
    beta_end=0.02,
)


# Fetch some data
x = next(iter(test_dataloader))
x = x[:batch_size].to(autocast_device)

# Corrupt with a range of amounts
timestep = torch.randint(0, max_timesteps, (x.shape[0],)).to(autocast_device)
noise = torch.randn_like(x).to(autocast_device)
noised_x = noise_scheduler.add_noise(x, noise, timestep)

# Get the model predictions
model.eval()
with torch.no_grad():
    pred = model(noised_x, timestep)

# Normalize for visualization - Subtracting the predicted from the noised input to obtain original image
denoised = noised_x - pred
denoised_vis = (denoised - denoised.min()) / (denoised.max() - denoised.min() + 1e-8)


# Plot
fig, axs = plt.subplots(4, 1, figsize=(12, 8))
axs[0].set_title('Input data', fontsize=8)
input_image = torchvision.utils.make_grid(x).cpu().clip(0, 1)
axs[0].imshow(input_image.permute(1,2,0).numpy().astype(np.float32))

axs[1].set_title(f'Corrupted data with timestep: {timestep.cpu().numpy()}', fontsize=8)
corrupted_image = torchvision.utils.make_grid(noised_x).cpu()
corrupted_image = corrupted_image.clamp(0, 1)
axs[1].imshow(corrupted_image.permute(1,2,0).numpy().astype(np.float32))

axs[2].set_title('Noise Predictions', fontsize=8)
predicted_noise = torchvision.utils.make_grid(pred.detach().cpu()).clip(0, 1)
axs[2].imshow((predicted_noise.permute(1,2,0).numpy().astype(np.float32)))

axs[3].set_title('Network Predictions', fontsize=8)
predicted_image = torchvision.utils.make_grid(denoised_vis.cpu()).clip(0, 1)
axs[3].imshow((predicted_image.permute(1,2,0).numpy().astype(np.float32)))

plt.subplots_adjust(hspace=0.4)  # vertical spacing
if Test:
    plt.savefig(f"figures/eval_{batch_size}_MNIST.png")
else:
    plt.savefig(f"figures/eval_{batch_size}_custom.png")

plt.show()