import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn

from diffusers import DDPMScheduler
from model import BasicUNet

from torch.amp import autocast

if torch.cuda.is_available():
    autocast_device = 'cuda'
    
else:
    autocast_device = 'cpu'


# Load example dataset for testing
dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Create the UNET model
model = BasicUNet(in_channels=1, out_channels=1).to(autocast_device)

# Load the trained model weights
model.load_state_dict(torch.load('model.pth', weights_only=True, map_location=autocast_device))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss() 

max_timesteps = 10

# Configurate the noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=max_timesteps,
    beta_start=0.0001,
    beta_end=0.02,
)

# Fetch some data
x, y = next(iter(train_dataloader))
x = x[:8].to(autocast_device) # Only using the first 8 for easy plotting

# Corrupt with a range of amounts
timestep = torch.randint(0, max_timesteps, (x.shape[0],)).to(autocast_device)
noise = torch.randn_like(x)
noised_x = noise_scheduler.add_noise(x, noise, timestep)

# Get the model predictions
model.eval()
with torch.no_grad():
    preds = model(noised_x, timestep)

# Normalize for visualization - Subtracting the predicted from the noised input to obtain original image
denoised = noised_x - preds
denoised_vis = (denoised - denoised.min()) / (denoised.max() - denoised.min() + 1e-8)
# Plot
fig, axs = plt.subplots(3, 1, figsize=(10, 7))
axs[0].set_title('Input data')
axs[0].imshow(torchvision.utils.make_grid(x)[0].cpu().clip(0, 1), cmap='Greys')
axs[1].set_title(f'Corrupted data with timestep: {timestep.cpu().numpy()}')
axs[1].imshow(torchvision.utils.make_grid(noised_x)[0].cpu().clip(0, 1), cmap='Greys')
axs[2].set_title('Network Predictions')
axs[2].imshow(torchvision.utils.make_grid(denoised_vis.cpu())[0], cmap='Greys')
plt.show()