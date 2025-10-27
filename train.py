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
# -----------------------------------------------
import os
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

# *Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet on MNIST or custom dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--max_timesteps", type=int, default=10, help="Number of timesteps")
    parser.add_argument("--test", action="store_true", help="Use MNIST test dataset")
    parser.add_argument("--model", type=str, default="UNET", help="Model type: UNET or Basic")
    parser.add_argument("--model_name", type=str, default=None, help="Checkpoint model name")
    parser.add_argument("--custom_model_name", type=str, default="model", help="Custom model name for saving")
    return parser.parse_args()
# ----------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    batch_size = args.batch_size
    num_epochs = args.epochs
    Test = args.test
    max_timesteps = args.max_timesteps
    model_name = args.model_name if (args.model_name and os.path.exists(f'models/checkpoints/{args.model_name}.pth')) else ''
    if not model_name:
        print(f"⚠️ Warning: Model '{args.model_name}' not found in models/checkpoints/. Starting from scratch.")
    custom_model_name = args.custom_model_name


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
        model = BasicUNet(in_channels=1, out_channels=1).to(autocast_device)
    else:
        model = UNET(in_channels = 1 if Test else 3, out_channels = 1 if Test else 3).to(autocast_device)
    
    model.load_state_dict(torch.load(f'models/{model_name}.pth', weights_only=True)) if model_name != '' else None

    print(f"\nInput channels:  {next(iter(train_dataloader)).size()}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss() 

    # *Configurate the noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=max_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
    )

    save_dir = "./models/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    best_loss = float('inf')

    # *Training loop
    for epoch in trange(num_epochs):
        model.train()
        for x in train_dataloader:
            x = x.to(autocast_device) 
            timestep = torch.randint(0, max_timesteps, (x.shape[0],)).to(autocast_device)
            noise = torch.rand_like(x).to(autocast_device)

            # *Add noise to the images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(x, noise, timestep).to(autocast_device)
            
            # *Predict the noise using the model - the model learns to denoise
            with autocast(device_type=autocast_device):
                optimizer.zero_grad()
                noise_pred = model(noisy_images, timestep) # (16, 3, 128, 128) or (8, 1, 28, 28)
                loss = loss_fn(noise_pred, noise)

            # *Backpropagate the loss and update the model parameters
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_loss = loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(save_dir, f"{custom_model_name}_epoch_{'test' if Test else ''}_{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss": loss.item()
            }, checkpoint_path)
            print(f"Model checkpoint saved to: {checkpoint_path}")
        
        # Validation loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x in val_dataloader:
                x = x.to(autocast_device)
                timestep = torch.randint(0, max_timesteps, (x.shape[0],)).to(autocast_device)
                noise = torch.rand_like(x).to(autocast_device)

                noisy_images = noise_scheduler.add_noise(x, noise, timestep).to(autocast_device)

                with autocast(device_type=autocast_device):
                    noise_pred = model(noisy_images, timestep)
                    val_loss = loss_fn(noise_pred, noise)

                val_losses.append(val_loss.item())
        current_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {current_loss:.4f}")
        


        # * Find the best model based on loss
        if current_loss < best_loss:
            best_loss = current_loss
            best_loss = current_loss
            best_model_path = os.path.join(save_dir, f"best_{'test' if Test else 'custom'}_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"🌟 New best model saved with loss {best_loss:.4f}")

    print("Training complete.")

    #TODO: Train for 200 epochs and save model checkpoints every 50 epochs
    #TODO: Finalize validation loop to monitor overfitting
    #TODO: Implement learning rate scheduler for better convergence
    #TODO: Load checkpoints if available to resume training

    # Save the model after training
    torch.save(model.state_dict(), f'./models/{custom_model_name}_{'test' if Test else ''}.pth') 