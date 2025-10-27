import torch
import torchvision
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from diffusers import DDPMScheduler
from model import BasicUNet, UNET
from data import get_dataset
from utils import collate_fn, sample_images
import argparse
import sys
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autocast_device = "cuda" if device.type == "cuda" else "cpu"

# *Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet on MNIST or custom dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_timesteps", type=int, default=1000, help="Number of timesteps")
    parser.add_argument("--test", action="store_true", help="Use MNIST test dataset")
    parser.add_argument("--checkpoint", action="store_true", help="Use a checkpoint model to eval")
    parser.add_argument("--model", type=str, default="UNET", help="Model type: UNET or Basic")
    parser.add_argument("--model_name", type=str, default="model", help="Custom model name for saving")
    return parser.parse_args()
# ----------------------------------------------
args = parse_args()
batch_size = args.batch_size
max_timesteps = args.max_timesteps
Test = args.test
Checkpoint = args.checkpoint
model_name = args.model_name
if Checkpoint:
        if not args.model_name:
            print("Warning: --checkpoint set but no --model_name provided. Exiting.")
            sys.exit(1)

        ckpt_path = f"models/checkpoints/{args.model_name}.pth"
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint file {ckpt_path} does not exist. Exiting.")
            sys.exit(1)
        model_name = args.model_name

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
    mnist_train = torchvision.datasets.MNIST(root="mnist/", train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(root="mnist/", train=False, download=True)

    # small validation split from train for completeness
    val_size = int(len(mnist_train) * 0.2)
    train_size = len(mnist_train) - val_size
    train, val = random_split(mnist_train, [train_size, val_size])


    train_dataloader = DataLoader(train, batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataloader = DataLoader(val, batch_size, collate_fn=collate_fn, shuffle=True)
    test_dataloader = DataLoader(mnist_test, batch_size, collate_fn=collate_fn, shuffle=False)


# Create the UNET model
in_ch = 1 if Test else 3
out_ch = 1 if Test else 3

model = BasicUNet(in_channels=in_ch, out_channels=out_ch).to(device) if args.model == "Basic" \
        else UNET(in_channels=in_ch, out_channels=out_ch).to(device)

if Test:
    # Load the trained model weights
    model.load_state_dict(torch.load(f"models/{model_name}_test.pth", weights_only=True))
else:
    # Load the trained model weights
    model.load_state_dict(torch.load(f"models/{model_name}.pth", weights_only=True))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss() 

# Configurate the noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=max_timesteps,
    beta_start=0.0001,
    beta_end=0.02,
)
# For diffusers sampling, set timesteps for inference (optionally different inference steps)
noise_scheduler.set_timesteps(max_timesteps)   # important

T = noise_scheduler.config.num_train_timesteps

# fetch a batch from test dataloader
batch = next(iter(test_dataloader))
# dataloader may return (x,y) or x directly
if isinstance(batch, (list, tuple)):
    x = batch[0]
else:
    x = batch
# ensure tensor and move to device
x = x.to(device)

# sample timesteps, noise and corrupt images
timestep = torch.randint(0, T, (x.shape[0],), device=device, dtype=torch.long)
noise = torch.randn_like(x, device=device)
noised_x = noise_scheduler.add_noise(x, noise, timestep)

# prepare betas / alphas
T = noise_scheduler.config.num_train_timesteps
betas = torch.linspace(noise_scheduler.config.beta_start, noise_scheduler.config.beta_end, T, dtype=torch.float32, device=device)
alphas = 1.0 - betas
alpha_cumprod = torch.cumprod(alphas, dim=0)

# Get the model predictions
model.eval()
with torch.no_grad():
    pred = model(noised_x, timestep)

# reconstruct x0 estimate from predicted noise
alpha_cumprod_t = alpha_cumprod[timestep].view(-1, 1, 1, 1)
sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)

# avoid division by zero
x0_pred = (noised_x - sqrt_one_minus_alpha_cumprod_t * pred) / (sqrt_alpha_cumprod_t + 1e-8)

# clamp for visualization in [0,1]
denoised_vis = x0_pred.clamp(0.0, 1.0).cpu()

# generate samples from pure noise (utils.sample_images should return CPU tensor; handle both)
samples = sample_images(model, noise_scheduler, img_size=28, device=device, n=16, Test=Test)
if isinstance(samples, np.ndarray):
    samples = torch.from_numpy(samples).float()
if isinstance(samples, torch.Tensor):
    # ensure channel-first (N,C,H,W)
    if samples.ndim == 4 and samples.shape[-1] in (1, 3):
        # HWC -> CHW
        samples = samples.permute(0, 3, 1, 2)
    samples = samples.cpu().clamp(0.0, 1.0)
    samples = samples.clip(0.0, 1.0)
else:
    raise TypeError("sample_images must return a numpy array or torch.Tensor")

def tensor_grid_to_numpy(tensor, nrow=8, rescale=True):
    """Make a torchvision grid and return a float32 numpy array.
    If rescale=True normalize to [0,1] using min/max. If rescale=False clip to [0,1]."""
    # Accept either torch.Tensor or numpy array
    if isinstance(tensor, np.ndarray):
        arr = tensor
        # If HWC -> CHW expectation already handled outside; ensure float32
        arr = arr.astype(np.float32)
        # If array is HWC single-channel -> squeeze last dim
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        # Rescale or clip
        mn, mx = float(arr.min()), float(arr.max())
        if rescale:
            if mn < 0.0 or mx > 1.0:
                if mx - mn > 1e-8:
                    arr = (arr - mn) / (mx - mn)
                else:
                    arr = np.clip(arr, 0.0, 1.0)
        else:
            arr = np.clip(arr, 0.0, 1.0)
        return arr
# torch.Tensor path
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=False, pad_value=1)
    # move to CPU, ensure float32
    grid = grid.detach().cpu().to(torch.float32)
    arr = grid.permute(1, 2, 0).numpy().astype(np.float32)
    # if single-channel, squeeze last dim
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    # Rescale to [0,1] if necessary
    mn, mx = float(arr.min()), float(arr.max())
    if rescale:
        if mn < 0.0 or mx > 1.0:
            if mx - mn > 1e-8:
                arr = (arr - mn) / (mx - mn)
            else:
                arr = np.clip(arr, 0.0, 1.0)
    else:
        arr = np.clip(arr, 0.0, 1.0)
    return arr

# ensure samples is a CPU float32 tensor
samples = samples.cpu().float()

# Debug print (remove later)
print("samples raw min/max/mean/std:", float(samples.min()), float(samples.max()), float(samples.mean()), float(samples.std()))

# Auto-detect range and map for plotting:
smin, smax = float(samples.min()), float(samples.max())
if smin >= 0.0 and smax <= 1.0:
    # already in [0,1], nothing to do
    samples_vis = samples.clamp(0.0, 1.0)
elif smin >= -1.5 and smax <= 1.5:
    # likely in [-1,1] -> map to [0,1]
    samples_vis = ((samples + 1.0) / 2.0).clamp(0.0, 1.0)
elif smax > 1.0 and smax <= 255.0:
    # integer image range 0-255 -> scale to [0,1]
    samples_vis = (samples / 255.0).clamp(0.0, 1.0)
else:
    # fallback: min/max normalize (keeps contrast)
    samples_vis = (samples - smin) / (smax - smin + 1e-8)
    samples_vis = samples_vis.clamp(0.0, 1.0)

# then continue to plotting using samples_vis
grid_arr = tensor_grid_to_numpy(samples_vis, nrow=4)
plt.figure(figsize=(6, 6))
plt.title("Generated Samples from Pure Noise", fontsize=14)
plt.imshow(grid_arr, cmap='gray' if in_ch == 1 else None)
plt.axis("off")
plt.savefig(f"figures/eval_generate_sample_{batch_size}_{'MNIST' if Test else 'custom'}.png", bbox_inches="tight")
plt.show()
plt.close()

# Also map denoised_vis similarly if it was computed in [-1,1]:
# denoised_vis = ((denoised_vis + 1.0) / 2.0).clamp(0.0, 1.0)

# Debug prints: inspect tensors before visualization
print("model device:", next(model.parameters()).device)
print("noised_x stats:", tuple(float(x) for x in (noised_x.min(), noised_x.max(), noised_x.mean(), noised_x.std())))
print("pred stats:", tuple(float(x) for x in (pred.min(), pred.max(), pred.mean(), pred.std())))
print("x0_pred stats:", tuple(float(x) for x in (x0_pred.min(), x0_pred.max(), x0_pred.mean(), x0_pred.std())))
print("samples stats (after sampling):", tuple(float(x) for x in (samples.min(), samples.max(), samples.mean(), samples.std())))
# Also print a small sample of pixel values
print("samples[0] first 10 pixels:", samples.view(samples.size(0), -1)[0, :10].cpu().numpy())

# --- debug checks: predict-type + scheduler consistency ---
print("noise_scheduler.config.num_train_timesteps:", noise_scheduler.config.num_train_timesteps)
print("scheduler.timesteps (len):", len(list(noise_scheduler.timesteps)), "first/last:", list(noise_scheduler.timesteps)[:3], list(noise_scheduler.timesteps)[-3:])

# compute true_noise for the batch (we have original x and noised_x)
alpha_cumprod_t = alpha_cumprod[timestep].view(-1,1,1,1)
sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
true_noise = (noised_x - sqrt_alpha_cumprod_t * x) / (sqrt_one_minus_alpha_cumprod_t + 1e-8)

mse_eps = torch.mean((pred - true_noise).pow(2)).item()
mse_x0 = torch.mean((pred - x).pow(2)).item()
print(f"MSE(pred, true_eps) = {mse_eps:.6g}, MSE(pred, x_clean) = {mse_x0:.6g}")

# sanity: print small sample comparisons
print("true_noise[0] first 8:", true_noise.view(true_noise.size(0), -1)[0,:8].cpu().numpy())
print("pred[0] first 8:", pred.view(pred.size(0), -1)[0,:8].cpu().numpy())
print("x0_pred[0] first 8:", x0_pred.view(x0_pred.size(0), -1)[0,:8].cpu().numpy())

# Single-step generation test (debug)
with torch.no_grad():
    t = torch.tensor([max_timesteps - 1], device=device, dtype=torch.long)  # use last training timestep
    x = torch.randn((1, in_ch, 28, 28), device=device)
    eps = model(x, t)
    # reuse alpha_cumprod from above
    alpha_t = alpha_cumprod[t].view(-1,1,1,1)
    sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_t)
    sqrt_alpha = torch.sqrt(alpha_t)
    x0_single = (x - sqrt_one_minus_alpha * eps) / (sqrt_alpha + 1e-8)
    # map for display (auto-detect range)
    x0v = x0_single.cpu()
    if (x0v.min() < -0.5) and (x0v.max() <= 1.5):
        x0v = ((x0v + 1.0) / 2.0).clamp(0,1)
    else:
        x0v = x0v.clamp(0,1)
    print("single-step x0 stats:", float(x0v.min()), float(x0v.max()), float(x0v.mean()), float(x0v.std()))
    torchvision.utils.save_image(x0v, "figures/debug_single_step_x0.png", nrow=1)

