import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn

from diffusers import DDPMScheduler
from model import BasicUNet, UNET
from data import get_dataset
from utils import load_batch_to_tensor, plot_images
from torch.amp import autocast
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


if args.test == False:
    print("Testing on custom dataset")
    dataset = get_dataset()
    dataset = load_batch_to_tensor(dataset, args.batch_size)
    train_dataloader = DataLoader(dataset, args.batch_size)
else:
    # Load example dataset for testing
    dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# Create the UNET model
if args.model == "Basic": model = BasicUNet(in_channels= 1 if args.test else 3, out_channels=1 if args.test else 3).to(autocast_device)
else: model = UNET(in_channels= 1 if args.test else 3, out_channels=1 if args.test else 3).to(autocast_device)

if args.test:
    # Load the trained model weights
    model.load_state_dict(torch.load('models/model_test.pth', weights_only=True))
else:
    # Load the trained model weights
    model.load_state_dict(torch.load('models/model.pth', weights_only=True))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss() 

max_timesteps = args.max_timesteps

# Configurate the noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=max_timesteps,
    beta_start=0.0001,
    beta_end=0.02,
)



# Fetch some data
x = next(iter(train_dataloader))
x = x[:args.batch_size].to(autocast_device)

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
fig, axs = plt.subplots(3, 1, figsize=(10, 7))
axs[0].set_title('Input data')
input_image = torchvision.utils.make_grid(x).cpu().clip(0, 1)
axs[0].imshow(input_image.permute(1,2,0).numpy())

axs[1].set_title(f'Corrupted data with timestep: {timestep.cpu().numpy()}')
corrupted_image = torchvision.utils.make_grid(noised_x).cpu()
corrupted_image = corrupted_image.clamp(0, 1)
axs[1].imshow(corrupted_image.permute(1,2,0).numpy())

axs[2].set_title('Network Predictions')
predicted_image = torchvision.utils.make_grid(denoised_vis.cpu()).clip(0, 1)
axs[2].imshow((predicted_image.permute(1,2,0).numpy()))
plt.show()