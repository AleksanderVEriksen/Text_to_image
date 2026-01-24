import torch, torchvision, argparse, os, sys, numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from diffusers import DDPMScheduler
from model import BasicUNet, UNET
from utils import ( 
    sample_images, 
    tensor_grid_to_numpy, 
    normalize_per_sample, 
    load_model_weights, 
    timesteps_to_str,  
    set_global_seed,
    load_data_from_dataset
)
from ema import ExponentialMovingAverage
import json, os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autocast_device = "cuda" if device.type == "cuda" else "cpu"

# *Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet on MNIST or custom dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "custom"], help="Dataset name")
    parser.add_argument("--max_timesteps", type=int, default=1000, help="Number of timesteps")
    parser.add_argument("--checkpoint", action="store_true", help="Use a checkpoint model to eval")
    parser.add_argument("--EMA", action="store_true", help="Use EMA weights for evaluation")
    parser.add_argument("--model", type=str, default="UNET", help="Model type: UNET or Basic", choices=['UNET', 'Basic'])
    parser.add_argument("--model_name", type=str, default="model", help="Custom model name for saving")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes for label embedding")
    parser.add_argument("--Augment", action="store_true", help="Use data augmentation")
    parser.add_argument("--guidance_scale", type=float, default=None)
    return parser.parse_args()
# ----------------------------------------------
args = parse_args()
batch_size = args.batch_size
max_timesteps = args.max_timesteps
Dataset = args.dataset
Checkpoint = args.checkpoint
model_name = args.model_name
num_classes = args.num_classes
Augment = args.Augment
EMA = args.EMA
img_size = 32 if Dataset == "mnist" else 64
set_global_seed(42)

def load_config(batch_size):
    path = f"models/{args.dataset}/{batch_size}/config.json"
    return json.load(open(path)) if os.path.isfile(path) else {}

# ----------------------------------------------
if Checkpoint:
        if not args.model_name:
            print("Warning: --checkpoint set but no --model_name provided. Exiting.")
            sys.exit(1)

        ckpt_path = f"models/{args.dataset}/checkpoints/{batch_size}/{args.model_name}.pth"
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint file {ckpt_path} does not exist. Exiting.")
            sys.exit(1)
        model_name = args.model_name
if EMA:
        if not args.model_name:
            print("Warning: --EMA set but no --model_name provided. Exiting.")
            sys.exit(1)

        ckpt_path = f"models/{args.dataset}/{batch_size}/{args.model_name}.pth"
        if not os.path.exists(ckpt_path):
            print(f"Warning: EMA file {ckpt_path} does not exist. Exiting.")
            sys.exit(1)
        model_name = args.model_name

# Load dataset from data.py
train_dataloader, val_dataloader, test_dataloader = load_data_from_dataset(Dataset, batch_size, Augment, verbose=False)

# Create the UNET model
in_ch = 1 if Dataset == "mnist" else 3

model = BasicUNet(in_channels=in_ch, num_classes=num_classes).to(device) if args.model == "Basic" \
    else UNET(in_channels=in_ch, num_classes=num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss() # L2 loss for noise prediction
scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
ema = ExponentialMovingAverage(model, decay=0.9999)


# After parsing args and before model init:
cfg = load_config(args.batch_size)
if cfg:
    img_size = cfg.get("img_size", 32)
    max_timesteps = cfg.get("max_timesteps", args.max_timesteps)
else:
    img_size = 32
    max_timesteps = args.max_timesteps

# Use cfg-driven max_timesteps for scheduler if needed:
# noise_scheduler = DDPMScheduler(num_train_timesteps=max_timesteps, ...)

# Then use it (replace both loading blocks with):
checkpoint = load_model_weights(
    model,
    model_name=model_name,
    dataset=Dataset,
    batch_size=batch_size,
    device=device,
    use_checkpoint=Checkpoint,
    use_ema=EMA
)
if checkpoint is not None and isinstance(checkpoint, dict):
    opt_state = checkpoint.get("optimizer_state_dict", None)
    if isinstance(opt_state, dict) and "param_groups" in opt_state:
        optimizer.load_state_dict(opt_state)
    else:
        print("Optimizer state not found in checkpoint; skipping optimizer.load_state_dict")
    ema_state = checkpoint.get("ema_state", None)
    if isinstance(ema_state, dict) and ema_state:
        try:
            ema.load_state_dict(ema_state)
        except Exception as e:
            print(f"EMA state not compatible; skipping. Reason: {e}")

# Configurate the noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=max_timesteps,
    beta_schedule="squaredcos_cap_v2", # scaled_linear | squaredcos_cap_v2
    beta_start=0.0001,
    beta_end=0.02,
    clip_sample=True
)
# For diffusers sampling, set timesteps for inference (optionally different inference steps)
noise_scheduler.set_timesteps(max_timesteps)   # important

assert img_size == 32, "Mismatch: model trained on 32x32 expected."

# fetch a batch from test dataloader
batch = next(iter(test_dataloader))
# dataloader may return (x,y) or x directly
if isinstance(batch, (list, tuple)):
    x, labels = batch
else:
    x = batch
# ensure tensor and move to device
x = x.to(device)

# sample timesteps, noise and corrupt images
timestep = torch.randint(0, max_timesteps, (x.shape[0],), device=device, dtype=torch.long)
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
    # if test loader yields labels, pass them in; otherwise leave None
    labels_for_batch = None
    if isinstance(batch, (list, tuple)) and len(batch) > 1:
        labels_for_batch = labels
    pred = model(noised_x, timestep, labels_for_batch.to(device) if labels_for_batch is not None else None)

# reconstruct x0 estimate from predicted noise
alpha_cumprod_t = alpha_cumprod[timestep].view(-1, 1, 1, 1)
sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)

# avoid division by zero
x0_pred = (noised_x - sqrt_one_minus_alpha_cumprod_t * pred) / (sqrt_alpha_cumprod_t + 1e-8)

# clamp for visualization in [0,1]
denoised_vis = x0_pred.clamp(0.0, 1.0).cpu()
# generate samples from pure noise (utils.sample_images may return (samples, intermediates))
ema.apply_shadow(model)            # swap in EMA weights
samples, timesteps_used = sample_images(
    model, 
    noise_scheduler, 
    img_size=img_size, 
    device=device, 
    n=16, 
    labels=torch.arange(10).repeat(2)[:16],
    return_intermediates=False,
    )
ema.restore(model)                 # restore original weights after sampling

# ensure samples is on CPU
samples = samples.cpu()
timesteps_used = timesteps_used.cpu()

# If your training normalized images to [-1, 1], map back:
samples = ((samples + 1.0) / 2.0).clamp(0.0, 1.0)

# --- Plotting ---

# visualize predicted noise (map to 0..1 for display) and show compact stats in title
pred_vis = pred.cpu()
pred_min = float(pred_vis.min().item())
pred_max = float(pred_vis.max().item())
pred_vis = (pred_vis - pred_min) / (pred_max - pred_min + 1e-8)

# format timesteps safely (truncate if too long)
ts_str = timesteps_to_str(timestep.cpu())

# Create figure with 4 stacked rows
fig, axs = plt.subplots(4, 1, figsize=(10, 12))

axs[0].set_title('Input data(Before noising)', fontsize=10)
axs[0].imshow(tensor_grid_to_numpy(x.cpu(), nrow=min(8, x.shape[0])), cmap='gray' if in_ch == 1 else None)
axs[0].axis('off')

axs[1].set_title(f'Corrupted data\n(timesteps: {ts_str})', fontsize=8)
axs[1].imshow(tensor_grid_to_numpy(noised_x.cpu(), nrow=min(8, noised_x.shape[0]), rescale=False), cmap='gray')
axs[1].axis('off')

pred_vis = normalize_per_sample(pred)
axs[2].set_title(f'Noise Predictions\n(timesteps: {ts_str})', fontsize=8)
axs[2].imshow(tensor_grid_to_numpy(pred_vis, nrow=min(8, pred_vis.shape[0]), rescale=False), cmap='gray')
axs[2].axis('off')

axs[3].set_title('Network Predictions (reconstructed x0)', fontsize=8)
axs[3].imshow(tensor_grid_to_numpy(denoised_vis, nrow=min(8, denoised_vis.shape[0])), cmap='gray' if in_ch == 1 else None)
axs[3].axis('off')

plt.subplots_adjust(hspace=0.35)
os.makedirs(f"figures/{Dataset}/eval/{batch_size}", exist_ok=True)
plt.savefig(f"figures/{Dataset}/eval/{batch_size}/eval.png", bbox_inches="tight")
plt.show()
plt.close()


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


# Plot generated samples
grid_arr = tensor_grid_to_numpy(samples_vis, nrow=min(8, samples_vis.shape[0]))
plt.figure(figsize=(6, 6))
plt.title(f"Generated Samples from Pure Noise, label=7", fontsize=8)
plt.imshow(grid_arr, cmap='gray' if in_ch == 1 else None)
plt.axis("off")
plt.savefig(f"figures/{Dataset}/eval/{batch_size}/eval_generate_sample.png", bbox_inches="tight")
plt.show()
plt.close()


# Single-step generation test (debug)
with torch.no_grad():
    t = torch.tensor([max_timesteps - 1], device=device, dtype=torch.long)  # use last training timestep
    x = torch.randn((1, in_ch, img_size, img_size), device=device)
    # Add label for conditioning (e.g., generate digit 7)
    label = torch.tensor([7], device=device)  # Change number as needed
    eps = model(x, t, labels=label)
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
    os.makedirs(f"figures/{Dataset}/debug/{batch_size}", exist_ok=True)
    torchvision.utils.save_image(x0v, f"figures/{Dataset}/debug/{batch_size}/debug_single_step_x0.png", nrow=1)

# Generate fully denoised samples for the chosen label (e.g. 7)
samples_vis, timesteps_used = sample_images(
    model,
    noise_scheduler,
    img_size=img_size,
    device=device,
    n=16,
    labels=7 if Dataset=="mnist" else None,
    return_intermediates=False  # only final x0
)

# If guidance requested, run classifier-free guidance variant:
guidance_scale = args.guidance_scale
if guidance_scale is not None and Dataset=="mnist":
    with torch.no_grad():
        # duplicate conditional/unconditional passes handled inside model.forward when guidance_scale provided
        pass  # model already supports guidance when called with guidance_scale; adapt call if needed

# Build grid and save with explicit “denoised” wording
grid_arr = tensor_grid_to_numpy(samples_vis.cpu(), nrow=min(8, samples_vis.shape[0]))
subset_ts = timesteps_used[timesteps_used <= 300]
ts_str_subset = timesteps_to_str(subset_ts)

plt.figure(figsize=(6, 3))
plt.title(f"Denoised Samples (x0) for label=7\n(timesteps ≤300 listed: {ts_str_subset})", fontsize=8)
plt.imshow(grid_arr, cmap='gray' if samples_vis.shape[1] == 1 else None)
plt.axis('off')
os.makedirs(f"figures/{Dataset}/eval/{batch_size}", exist_ok=True)
plt.savefig(f"figures/{Dataset}/eval/{batch_size}/eval_generate_sample_denoised.png", bbox_inches='tight')
plt.close()


