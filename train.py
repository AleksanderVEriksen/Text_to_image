from diffusers import DDPMScheduler
from tqdm.auto import tqdm  # Replace existing tqdm import
from utils import collate_fn, sample_images, validate
from training import train_epoch
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
from ema import ExponentialMovingAverage
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
# TODO: Train for another 100 epochs on fresh model. Evaluate ema model also
# ----------------------------------------------
# *Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet on MNIST or custom dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--max_timesteps", type=int, default=1000, help="Number of timesteps")
    parser.add_argument("--test", action="store_true", help="Use MNIST test dataset")
    parser.add_argument("--model", type=str, default="UNET", help="Model type: UNET or Basic", choices=['UNET', 'Basic'])
    parser.add_argument("--num_classes", type=int, default=10, help="Number of label classes for label embedding")
    parser.add_argument("--checkpoint", action="store_true", help="Use a checkpoint to resume training")
    parser.add_argument("--model_name", type=str, default="model", help="Custom model name for saving")
    return parser.parse_args()
# ----------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    batch_size = args.batch_size
    num_epochs = args.epochs
    Test = args.test
    num_classes = args.num_classes
    max_timesteps = args.max_timesteps
    Checkpoint = args.checkpoint
    current_epoch = 0
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
            num_epochs = num_epochs - current_epoch
        else:
            current_epoch = 0
        
    else:
        if not os.path.exists('models'):
            print("No models directory found. Training from scratch.")
            os.makedirs('models', exist_ok=True)
            model_name = args.model_name
        if os.path.exists('model') and not os.path.listdir('models'):
            print("Models directory is empty. Training from scratch.")
            model_name = "model"
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
        model = BasicUNet(in_channels=1, out_channels=1, num_classes=num_classes).to(device)
    else:
        model = UNET(in_channels = 1 if Test else 3, out_channels = 1 if Test else 3, num_classes=num_classes).to(device)
    

    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=5e-5,
                                weight_decay=0.01,
                                betas=(0.9, 0.999))
    loss_fn = nn.MSELoss()
    scheduler_lr = lr_scheduler.CosineAnnealingLR(
                                    optimizer, 
                                    T_max=num_epochs,
                                    eta_min=1e-6)
    best_loss = float('inf')
    
    save_every_epochs = 5
    sample_every_epochs = 2

    ema = ExponentialMovingAverage(model, decay=0.9999)

    if Checkpoint:
        ckpt_file = os.path.join("models", "checkpoints", f"{model_name}.pth")
        ckpt = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt.get("optimizer_state_dict", {}))
        scaler.load_state_dict(ckpt.get("scaler_state_dict", {}))
        ema.load_state_dict(ckpt.get("ema_state", {}))
        start_epoch = ckpt.get("epoch", 0)
        best_loss = ckpt.get("loss", best_loss)
    else:
        # if a pretrained stateless file is expected, map to device; skip if file missing
        weights_file = os.path.join("models", f"{model_name}_Batch_size_{batch_size}_Max_timesteps_{max_timesteps}{'_test' if Test else ''}.pth")
        if os.path.exists(weights_file):
            try:
                model.load_state_dict(torch.load(weights_file, map_location=device))
            except Exception:
                # ignore if shape mismatch or not a state_dict
                pass
        start_epoch = 0

    # show a sample batch shape (collate_fn returns (images, labels))
    sample = next(iter(train_dataloader))
    if isinstance(sample, (tuple, list)):
        images, labels = sample
        print(f"\nInput sample shape: {tuple(images.shape)}")
        print(f"Labels shape: {tuple(labels.shape)}\n")
    else:
        images = sample
        print(f"\nInput sample shape: {tuple(images.shape)}\n")
    

    # *Configurate the noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=max_timesteps,
        beta_schedule="scaled_linear",
        beta_start=0.0001,
        beta_end=0.02,
        clip_sample=True
    )

    noise_scheduler.set_timesteps(max_timesteps)

    save_dir = "./models/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    T = noise_scheduler.config.num_train_timesteps

    # *Training loop
    torch.autograd.set_detect_anomaly(False)   # enable True only when debugging
    global_step = 0
    val_losses = []
    running_avg_losses = []

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = train_epoch(
        model, train_dataloader, optimizer, scheduler_lr,
        noise_scheduler, loss_fn, device, scaler, epoch, num_epochs
    )
        # Step learning rate scheduler
        scheduler_lr.step()
        
        # Log average loss
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"\nEpoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
        
    # *Validation
        val_loss, running_avg_loss = validate(model, val_dataloader, noise_scheduler, loss_fn, device)
        
        # After dataset creation, add lists to store metrics
        val_losses = []
        running_avg_losses = []

        val_losses.append(val_loss)
        running_avg_losses.append(running_avg_loss)
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Training loss: {epoch_loss/len(train_dataloader):.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Running avg val loss: {running_avg_loss:.4f}")
        print(f"Best validation loss: {min(val_losses):.4f}")
        print("-" * 50)
        
        # Save if running average improved
        if running_avg_loss == min(running_avg_losses):
            print(f"Running average validation loss improved. Saving checkpoint...")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler_lr.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': val_loss,
                'running_avg_loss': running_avg_loss,
                'val_losses': val_losses,
                'running_avg_losses': running_avg_losses,
                'ema_state': ema.state_dict(),
            }, f"models/best_model.pth")

        
        if (epoch + 1) % save_every_epochs == 0:
            ckpt_path = os.path.join(save_dir, f"model_BS_{batch_size}_MaxT_{max_timesteps}{'_test' if Test else ''}_{current_epoch+epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if device.type == "cuda" else {},
                "ema_state": ema.state_dict(),
                "loss": running_avg_loss,
            }, ckpt_path)
        # sample and save images for quick inspection
        if epoch % sample_every_epochs == 0:
            model.eval()
            with torch.no_grad():
                ema.apply_shadow(model)
                generated_images = sample_images(
                    model, 
                    noise_scheduler, 
                    img_size=28 if Test else 64, 
                    device=device, 
                    n=16, 
                    Test=True, 
                    labels= torch.arange(4).repeat(4),
                    num_classes=num_classes)
                ema.restore(model)
                # sample_images may return (samples, intermediates) or a tensor/ndarray
                if isinstance(generated_images, (list, tuple)):
                    generated_images = generated_images[0]
            # save grid
            os.makedirs("figures/samples", exist_ok=True)
            torchvision.utils.save_image(generated_images, f"figures/samples/samples_epoch_{epoch}.png", nrow=4, normalize=True)

    print(f"\nBest model saved with loss {best_loss:.4f}")
    print("\nTraining complete.")

    # Save EMA state plus a copy of the model with EMA weights applied
    os.makedirs(f"./models/Batch_size_{batch_size}", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "ema_state": ema.state_dict(),
    }, f"./models/Batch_size_{batch_size}/{model_name}{'_test' if Test else ''}_{num_epochs}.pth")


    # Also save a separate file with EMA weights applied to the model (for easy inference)
    os.makedirs("./models/EMA", exist_ok=True)
    ema.apply_shadow(model)
    torch.save(model.state_dict(), f"./models/EMA/{model_name}_EMA_BS_{batch_size}{'_test' if Test else ''}.pth")
    ema.restore(model)
    print(f"EMA model saved as ./models/EMA/{model_name}_EMA_BS_{batch_size}{'_test' if Test else ''}.pth")