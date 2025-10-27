from diffusers import DDPMScheduler
from utils import collate_fn
# -----------------------------------------------
from torch.utils.data import DataLoader, random_split
import torchvision
# -----------------------------------------------
import torch
import sys
import torch.nn as nn
import torch, gc
import torch.optim.lr_scheduler as lr_scheduler
# -----------------------------------------------
import os
import re
# -----------------------------------------------
from model import UNET, BasicUNet
from tqdm import trange
# -----------------------------------------------
# *Automatic Mixed Precision - saves memory and speeds up training
from torch.amp import autocast, GradScaler
import argparse
from data import get_dataset
# Clear cache
gc.collect()
torch.cuda.empty_cache()
# -----------------------------------------------
# determine device once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autocast_device = "cuda" if device.type == "cuda" else "cpu"
# ----------------------------------------------
# Sets scaler
scaler = GradScaler()
# ----------------------------------------------

# *Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet on MNIST or custom dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--max_timesteps", type=int, default=1000, help="Number of timesteps")
    parser.add_argument("--test", action="store_true", help="Use MNIST test dataset")
    parser.add_argument("--model", type=str, default="UNET", help="Model type: UNET or Basic")
    parser.add_argument("--checkpoint", action="store_true", help="Use a checkpoint to resume training")
    parser.add_argument("--model_name", type=str, default="model", help="Custom model name for saving")
    return parser.parse_args()
# ----------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    batch_size = args.batch_size
    num_epochs = args.epochs
    Test = args.test
    max_timesteps = args.max_timesteps
    Checkpoint = args.checkpoint
    if Checkpoint:
        if not args.model_name:
            print("Warning: --checkpoint set but no --model_name provided. Exiting.")
            sys.exit(1)

        ckpt_path = f"models/checkpoints/{args.model_name}.pth"
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint file {ckpt_path} does not exist. Exiting.")
            sys.exit(1)
        model_name = args.model_name

        m = re.search(r'_(\d+)\.pth|_(\d+)$', model_name)
        if m:
            current_epoch = int(m.group(1) or m.group(2))
            print(f"Resuming training from epoch {current_epoch}")
        else:
            current_epoch = 0
        
    else:
        model_name = args.model_name


    # *Load dataset from data.py
    if args.test == False:
        print("Training on custom dataset")

        train = get_dataset(train = True)
        test = get_dataset(test = True)
        val = get_dataset(val = True)

        train_dataloader = DataLoader(train, batch_size, collate_fn=collate_fn)
        val_dataloader = DataLoader(val, batch_size, collate_fn=collate_fn)
        test_dataloader = DataLoader(test, batch_size, collate_fn=collate_fn)

    else:
        # *Load example dataset for testing
        print("\n---Testing on MNIST dataset---")
        train_ = torchvision.datasets.MNIST(root="mnist/", train=True, download=True)
        test_ = torchvision.datasets.MNIST(root="mnist/", train=False, download=True)

        train_size = int((1 - len(train_)*0.8))
        val_size = (len(train_)-train_size)
        train, val = random_split(train_, [train_size, val_size] )

        train_dataloader = DataLoader(train, batch_size, collate_fn=collate_fn)
        val_dataloader = DataLoader(val, batch_size, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_, batch_size, collate_fn=collate_fn)

    # *Create the UNET model
    if args.model == "Basic":
        model = BasicUNet(in_channels=1, out_channels=1).to(device)
    else:
        model = UNET(in_channels = 1 if Test else 3, out_channels = 1 if Test else 3).to(device)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    scheduler_lr = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    best_loss = float('inf')

    if Checkpoint:
        ckpt_file = os.path.join("models", "checkpoints", f"{model_name}{'_test' if Test else ''}_{current_epoch}.pth")
        ckpt = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt.get("optimizer_state_dict", {}))
        scaler.load_state_dict(ckpt.get("scaler_state_dict", {}))
        start_epoch = ckpt.get("epoch", 0)
        best_loss = ckpt.get("loss", best_loss)
    else:
        # if a pretrained stateless file is expected, map to device; skip if file missing
        weights_file = os.path.join("models", f"{model_name}{'_test' if Test else ''}.pth")
        if os.path.exists(weights_file):
            try:
                model.load_state_dict(torch.load(weights_file, map_location=device))
            except Exception:
                # ignore if shape mismatch or not a state_dict
                pass
        start_epoch = 0

    print(f"\nInput channels:  {next(iter(train_dataloader)).size()}\n")
    

    # *Configurate the noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=max_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
    )

    noise_scheduler.set_timesteps(max_timesteps)

    save_dir = "./models/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    T = noise_scheduler.config.num_train_timesteps

    # *Training loop
    for epoch in trange(num_epochs):
        model.train()
        epoch_losses = []
        for batch in train_dataloader:
            # support dataloader returning (x, y) or just x
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)

            timestep = torch.randint(0, T, (x.shape[0],), device=device, dtype=torch.long)
            noise = torch.randn_like(x, device=device)
            # *Add noise to the images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(x, noise, timestep)
            # *Predict the noise using the model - the model learns to denoise
            optimizer.zero_grad()
            if device.type == "cuda":
                # mixed precision only on CUDA
                with autocast("cuda"):
                    noise_pred = model(noisy_images, timestep)
                    loss = loss_fn(noise_pred, noise)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # CPU path: no autocast/scaler
                noise_pred = model(noisy_images, timestep)
                loss = loss_fn(noise_pred, noise)
                loss.backward()
                optimizer.step()

            epoch_losses.append(loss.item())

        scheduler_lr.step()
        avg_train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        
        # Validation loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_dataloader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device)
                timestep = torch.randint(0, T, (x.shape[0],), device=device, dtype=torch.long)
                noise = torch.randn_like(x, device=device)    # gaussian noise
                noisy_images = noise_scheduler.add_noise(x, noise, timestep)

                if device.type == "cuda":
                    with autocast("cuda"):
                        noise_pred = model(noisy_images, timestep)
                        vloss = loss_fn(noise_pred, noise)
                else:
                    noise_pred = model(noisy_images, timestep)
                    vloss = loss_fn(noise_pred, noise)
                val_losses.append(vloss.item())
        avg_val_loss = sum(val_losses) / max(1, len(val_losses))
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

        # save best and periodic checkpoints
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_path = os.path.join(save_dir, f"best{'_test' if Test else ''}_model.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if device.type == "cuda" else {},
                "loss": best_loss
            }, best_model_path)

        if (epoch + 1) % 50 == 0:
            ckpt_path = os.path.join(save_dir, f"{model_name}{'_test' if Test else ''}_{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if device.type == "cuda" else {},
                "loss": avg_val_loss
            }, ckpt_path)

    print(f"\nBest model saved with loss {best_loss:.4f}")
    print("\nTraining complete.")

    # Save the model after training
    torch.save(model.state_dict(), f"./models/{model_name}{'_test' if Test else ''}.pth") 