from diffusers import DDPMScheduler
from training import train_epoch
import warnings
import time
# -----------------------------------------------
import torchvision
from torch.utils.data import Dataset as TorchDataset
# -----------------------------------------------
import torch
import sys
import torch.nn as nn
import torch, gc
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
# -----------------------------------------------
import os
import re
import numpy as np
# -----------------------------------------------
# MLflow for experiment tracking (if needed)
import mlflow
import platform
import traceback
from contextlib import nullcontext

def _end_mlflow_run(status: str) -> None:
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run(status=status)
    except Exception:
        # ignore end_run errors during shutdown
        pass

# -----------------------------------------------
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID")
# -----------------------------------------------
from model import UNET, BasicUNet
# -----------------------------------------------
# *Automatic Mixed Precision - saves memory and speeds up training
from torch.amp import GradScaler
import argparse
from ema import ExponentialMovingAverage
from utils import (
    set_global_seed,
    save_config,
    plot_fid,
    load_data_from_dataset,
    save_with_retry,
    sample_images,
    validate,
    plot_losses,
    build_signature,
    log_metrics_safe,
    log_model_safe,
    log_artifact_safe
    # Newly used / available helpers
)
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
# Suppress the LR scheduler deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")
# ----------------------------------------------
# TODO: Train for a total 500-2000 epochs on a model. Evaluate ema model also
# ----------------------------------------------
# *Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train UNET on MNIST or custom dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_timesteps", type=int, default=1000)
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "custom"])
    parser.add_argument("--model", type=str, default="UNET", choices=['UNET', 'Basic'])
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--model_name", type=str, default="model")
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--val_max_batches", type=int, default=32)
    parser.add_argument("--sample_every_epoch", type=int, default=50)
    parser.add_argument("--save_every_epoch", type=int, default=10)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--top_k_models", type=int, default=3)
    parser.add_argument("--fid_epoch_calc", type=int, default=20)
    parser.add_argument("--is_epoch_calc", type=int, default=20, help="Epoch interval to compute Inception Score")
    parser.add_argument("--no_ema_validate", action="store_true", help="Disable EMA weights during validation")
    parser.add_argument("--use_weighted_snr", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)  # added (used by set_global_seed)
    parser.add_argument("--disable_mlflow", action="store_true", help="Disable MLflow logging and system metrics")
    return parser.parse_args()
# ----------------------------------------------
if __name__ == "__main__":

    # * Parse arguments
    args = parse_args()
    batch_size = args.batch_size
    num_epochs = args.epochs
    num_classes = args.num_classes
    max_timesteps = args.max_timesteps
    Checkpoint = args.checkpoint
    Dataset = args.dataset
    Augment = args.augment
    FID_EPOCH_CALC = args.fid_epoch_calc
    IS_EPOCH_CALC = args.is_epoch_calc
    # * Initialize variables
    current_epoch = 0
    patience_counter = 0
    best_fid_path = ""
    best_loss = float('inf')
    best_val_loss = float('inf')
    best_fid = float('inf')  # Initialize best FID for early stopping
    img_size = 32 if Dataset == "mnist" else 64
    num_channels = 1 if Dataset == "mnist" else 3
    top_models = []

    train_dataloader = None
    val_dataloader = None
    test_dataloader = None

    # *Add existing models in the directory to the top_models list
    model_dir = f"./models/{args.dataset}/checkpoints/{args.batch_size}"
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith(".pth"):
                path = os.path.join(model_dir, filename)
                try:
                    ckpt = torch.load(path, map_location=device)
                    loss = ckpt.get("loss", None)
                    if loss is not None:
                        top_models.append((loss, path))
                except Exception as e:
                    print(f"Error loading checkpoint {path}: {str(e)}")

    # *Handle model naming and checkpoint resumption
    if Checkpoint:
        if not args.model_name:
            print("Warning: --checkpoint set but no --model_name provided. Exiting.")
            sys.exit(1)

        ckpt_path = f"models/{args.dataset}/checkpoints/{args.batch_size}/{args.model_name}.pth"
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
            # Use batch_size folders within the models directory
        batch_size_dir = f"./models/{args.dataset}/{args.batch_size}"
        if not os.path.exists(batch_size_dir):
            print(f"No models directory for batch_size {args.batch_size} found. Training from scratch.")
            os.makedirs(batch_size_dir, exist_ok=True)
        model_name = args.model_name


    # *Load dataset from data.py
    train_dataloader, val_dataloader, test_dataloader = load_data_from_dataset(Dataset, batch_size, Augment)
    
    
    # *Create the UNET model
    if args.model == "Basic":
        model = BasicUNet(in_channels=num_channels, num_classes=num_classes).to(device)
    else:
        model = UNET(in_channels=num_channels, num_classes=num_classes).to(device)

    # *Set up optimizer, loss function, and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=5e-5,
                                weight_decay=0.01,
                                betas=(0.9, 0.999))
    loss_fn = nn.MSELoss()
    scheduler_lr = CosineAnnealingLR(
                                    optimizer, 
                                    T_max=num_epochs,
                                    eta_min=1e-6)
    # Warmup for first 10 epochs
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)
    scheduler_lr = SequentialLR(optimizer, schedulers=[warmup_scheduler, scheduler_lr], milestones=[10])
    
    # *Set up Exponential Moving Average (EMA) for the model
    ema = ExponentialMovingAverage(model, decay=0.9999)

    if Checkpoint:
        ckpt_file = os.path.join("models", "checkpoints", f"{args.batch_size}", f"{model_name}.pth")
        ckpt = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt.get("optimizer_state_dict", {}))
        scaler.load_state_dict(ckpt.get("scaler_state_dict", {}))
        ema.load_state_dict(ckpt.get("ema_state", {}))
        start_epoch = ckpt.get("epoch", 0)
        best_loss = ckpt.get("loss", best_loss)
    else:
        # if a pretrained stateless file is expected, map to device; skip if file missing
        weights_file = os.path.join("models", f"{args.batch_size}", f"{model_name}{'_test' if Dataset == 'mnist' else ''}.pth")
        if os.path.exists(weights_file):
            try:
                model.load_state_dict(torch.load(weights_file, map_location=device))
            except Exception:
                # ignore if shape mismatch or not a state_dict
                pass
        start_epoch = 0

    # Ensure dataloaders are initialized
    if train_dataloader is None or val_dataloader is None or test_dataloader is None:
        raise RuntimeError("Dataloaders were not initialized; check dataset selection.")

    # *Show 10 samples in a grid for preview purposes
    # show a sample batch shape (collate_fn returns (images, labels))
    sample = next(iter(train_dataloader))
    if isinstance(sample, (tuple, list)):
        images, labels = sample
        print(f"\nInput sample shape: {tuple(images.shape)}")
        print(f"Labels shape: {tuple(labels.shape)}\n")
    else:
        images = sample
        print(f"\nInput sample shape: {tuple(images.shape)}\n")
    
    if isinstance(sample, (tuple, list)):
        num_preview = min(10, images.shape[0])
        preview_images = images[:num_preview].cpu()
        preview_labels = labels[:num_preview].cpu().tolist()

        preview_save_path = f"figures/{args.dataset}/preview/sample_images_grid_{args.dataset}.png"
        os.makedirs(f"figures/{args.dataset}/preview", exist_ok=True)
        try:
            torchvision.utils.save_image(
                preview_images,
                preview_save_path,
                nrow=num_preview // 2,
                normalize=True
            )
            print(f"Saved preview grid of {num_preview} images to {preview_save_path}\n")
        except Exception as e:
            print(f"Failed to save preview grid: {e}")
    # ----------------------------------------------

    # *Configurate the noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=max_timesteps,
        beta_schedule="scaled_linear",
        beta_start=0.0001,
        beta_end=0.02,
        clip_sample=True
    )

    noise_scheduler.set_timesteps(max_timesteps)

    check_save_dir = f"./models/{args.dataset}/checkpoints/{args.batch_size}"
    os.makedirs(check_save_dir, exist_ok=True)
    save_dir = f"./models/{args.dataset}/{args.batch_size}"
    os.makedirs(save_dir, exist_ok=True)

    T = noise_scheduler.config.num_train_timesteps

    # *Training loop
    torch.autograd.set_detect_anomaly(False)   # enable True only when debugging
    global_step = 0

    # Tracking lists for plotting (epochs separate from losses)
    train_epochs = []
    val_epoch_points = []
    # Already existing lists:
    val_losses = []
    train_losses = []
    running_avg_losses = []
    fid_scores = []
    fid_epochs = []

    # Set global seed
    set_global_seed(args.seed if hasattr(args, "seed") else 42)

    # After scheduler init, cache alphas_cumprod for SNR weighting
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

    # Save config once (removed non-existent args.model_type / args.ema_decay)
    save_config({
        "version": 1,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "model": args.model,
        "embedding_dim": 512,
        "num_classes": args.num_classes,
        "max_timesteps": args.max_timesteps,
    }, f"models/{args.dataset}/{args.batch_size}/config.json")

    train_epoch_losses = []  # per-epoch losses (for plotting)

    LOG_STATUS = not args.disable_mlflow  # Controlled by CLI flag

    # Configure MLflow (autolog + system metrics) only when enabled
    if LOG_STATUS:
        try:
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))
            mlflow.pytorch.autolog(log_models=False)
            # IMPORTANT: Enable system metrics monitoring
            try:
                mlflow.config.enable_system_metrics_logging()
                mlflow.config.set_system_metrics_sampling_interval(600)
            except Exception as e:
                # Older MLflow versions may not have config API
                print(f"MLflow system metrics config not available: {e}")
        except Exception as e:
            print(f"MLflow setup skipped: {e}")

    # *Start MLflow run for experiment tracking
    try:
        if LOG_STATUS and mlflow.active_run() is not None:
            mlflow.end_run()
        run_ctx = mlflow.start_run(run_name=f"{args.dataset}_bs{args.batch_size}_ep{num_epochs}", experiment_id=EXPERIMENT_ID, nested=True) if LOG_STATUS else nullcontext()
        with run_ctx as run:
            if LOG_STATUS:
                mlflow.log_params({
                "epochs": num_epochs,
                "batch_size": args.batch_size,
                "model": args.model,
                "num_classes": args.num_classes,
                "max_timesteps": args.max_timesteps,
                "dataset": args.dataset,
                "augment": args.augment,
                "seed": args.seed
                })
            # Log system info within the active run (prevents implicit extra runs)
            if LOG_STATUS:
                mlflow.log_params({
                    "device": device.type,
                    "os": platform.platform(),
                })
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    mlflow.log_params({
                        "gpu_name": torch.cuda.get_device_name(0),
                        "gpu_total_mem_bytes": props.total_memory,
                        "cuda_version": torch.version.cuda,
                        "cudnn_version": torch.backends.cudnn.version(),
                    })
            for epoch in range(start_epoch, num_epochs):
                current_epoch += 1
                # Print time
                start_time = time.time()
                # Default previous-best running loss for this epoch (updated after validation)
                prev_best_running = float('inf')
                # Initialize per-epoch optional metrics to ensure defined before checks
                epoch_is_score = None
                epoch_fid_score = None
                # train_epoch should already average over dataloader; if not, divide by len(train_dataloader)
                epoch_loss = train_epoch(
                    model, train_dataloader, noise_scheduler, optimizer,
                    loss_fn, device, ema=ema, mixed_precision=True, use_weighted_snr=True,
                    alphas_cumprod=alphas_cumprod
                )

                # * Validation step
                if (current_epoch) % args.val_every == 0:
                    # Use EMA weights for validation and optional FID sampling
                    use_ema_val = not args.no_ema_validate
                    if use_ema_val:
                        ema.apply_shadow(model)
                    try:
                        val_results = validate(
                            model,
                            current_epoch,
                            val_dataloader,
                            noise_scheduler,
                            loss_fn,
                            device,
                            max_batches=args.val_max_batches,
                            calculate_fid_score=True,
                            fid_epoch_calc=FID_EPOCH_CALC,
                            calculate_is_score=True,
                            is_epoch_calc=IS_EPOCH_CALC,
                            img_size=img_size
                        )
                    finally:
                        if use_ema_val:
                            ema.restore(model)
                    val_loss = val_results['val_loss']
                    # Compute an EMA for running_avg_loss across validation epochs
                    if running_avg_losses:
                        _prev_running = running_avg_losses[-1]
                        _beta = 0.9
                        running_avg_loss = _beta * _prev_running + (1 - _beta) * val_loss
                        prev_best_running = min(running_avg_losses)
                    else:
                        running_avg_loss = val_loss
                        prev_best_running = float('inf')
                    epoch_fid_score = val_results.get('fid_score', None)
                    epoch_is_score = val_results.get('is_score', None)

                    val_losses.append(val_loss)
                    running_avg_losses.append(running_avg_loss)
                    val_epoch_points.append(current_epoch)


                    if epoch_fid_score is not None:
                        fid_scores.append(epoch_fid_score)
                        fid_epochs.append(current_epoch)

                        if epoch_fid_score < best_fid:
                            best_fid = epoch_fid_score
                            patience_counter = 0
                            # apply EMA weights for saving best FID
                            if ema is not None:
                                ema.apply_shadow(model)
                            best_fid_path = f"{save_dir}/best_fid_model.pth"
                            save_with_retry(
                                best_fid_path,
                                {
                                    'epoch': epoch + 1,
                                    'best_fid': best_fid,
                                    'is_score': epoch_is_score,
                                    'val_loss': val_loss,
                                    'running_avg_loss': running_avg_loss,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'scheduler_state_dict': scheduler_lr.state_dict(),
                                    'scaler_state_dict': scaler.state_dict(),
                                    'ema_state': ema.state_dict() if ema is not None else None,
                                    'batch_size': batch_size,
                                    'version': 1
                                }
                            )
                            if ema is not None:
                                # optional: restore original weights after save
                                pass
                            print(f"Saved best FID model to {best_fid_path} (FID={epoch_fid_score:.4f})")
                        else:
                            patience_counter += 1
                        if patience_counter > args.patience:
                            print(f"Early stopping (FID patience {args.patience}) at epoch {current_epoch}")
                            break

        # ------------------------------------------------------------
                # Keep expensive image sampling less frequent
                if (current_epoch) % args.sample_every_epoch == 0 or (epoch == 0):
                    with torch.no_grad():
                        model.eval()
                        # Test reconstruction of specific digits
                        n_images = 16
                        test_labels = torch.full((n_images,), 7, device=device)  # Test multiple 7s
                        ema.apply_shadow(model)
                        generated_images = sample_images(
                            model, 
                            noise_scheduler, 
                            img_size=img_size, 
                            device=device, 
                            n=n_images, 
                            labels=test_labels,
                            )
                        ema.restore(model)
                        # sample_images may return (samples, intermediates) or a tensor/ndarray
                        if isinstance(generated_images, (list, tuple)):
                            generated_images = generated_images[0].cpu()
                        else:
                            generated_images = generated_images.cpu()
                    # save grid
                    os.makedirs(f"figures/{args.dataset}/samples/{args.batch_size}", exist_ok=True)
                    sample_save_path = f"figures/{args.dataset}/samples/{args.batch_size}/digit_7_epoch_{current_epoch}.png"
                    try:
                        torchvision.utils.save_image(
                            generated_images,
                            sample_save_path,
                            nrow=n_images // 4,
                            normalize=True
                        )
                    except Exception as e:
                        print(f"Failed to save generated samples grid: {e}")
        # -------------------------------------------------------------
                # * ------------------ Compute and log metrics ------------------ *
                
                avg_epoch_loss = float(epoch_loss)
                train_epoch_losses.append(avg_epoch_loss)
                train_epochs.append(current_epoch)

                # Log metrics to MLflow
                if (current_epoch) % args.val_every == 0:
                    log_metrics_safe({"val_loss": val_loss}, step=current_epoch, log_status=LOG_STATUS)
                if epoch_fid_score is not None and current_epoch % FID_EPOCH_CALC == 0:
                    log_metrics_safe({"fid_score": epoch_fid_score}, step=current_epoch, log_status=LOG_STATUS)
                if epoch_is_score is not None and current_epoch % IS_EPOCH_CALC == 0:
                    log_metrics_safe({"is_score": epoch_is_score}, step=current_epoch, log_status=LOG_STATUS)
                if (current_epoch % 10 == 0 or epoch == 0):
                    log_metrics_safe({"train_loss": avg_epoch_loss}, step=current_epoch, log_status=LOG_STATUS)
                    if running_avg_loss is not None:
                        log_metrics_safe({"running_avg_loss": running_avg_loss}, step=current_epoch, log_status=LOG_STATUS)
                
                # Print metrics
                print(f"\nEpoch {current_epoch}/{num_epochs}")
                print(f"Training loss: {avg_epoch_loss:.4f}")
                print(f'Best training loss so far: {min(train_epoch_losses):.4f}')
                if (current_epoch) % args.val_every == 0:
                    print(f"Validation loss: {val_loss:.4f}")
                    print(f"Running avg val loss: {running_avg_loss:.4f}")
                    print(f"Best validation loss: {min(val_losses):.4f}")
                    if epoch_fid_score is not None and current_epoch % FID_EPOCH_CALC == 0:
                        print(f"FID score: {epoch_fid_score:.4f}")
                        print(f"Best FID: {best_fid:.4f}")
                        print(f"Patience counter: {patience_counter}/{args.patience}")
                    if epoch_is_score is not None and current_epoch % IS_EPOCH_CALC == 0:
                        print(f"Inception Score: {epoch_is_score:.4f}")
                
                end_time = time.time()
                print(f"Time: {end_time - start_time:.2f} seconds")

                print("-" * 50)
                
                # * ------------------ Save model checkpoints ------------------ *
                if (current_epoch) % args.val_every == 0: 
                    # Save when the EMA running avg improves over previous best
                    if running_avg_loss < prev_best_running:
                        print("Running average validation loss improved. Saving checkpoint...")
                        best_model_path = f"{save_dir}/best_model.pth"
                        best_loss = running_avg_loss
                        save_with_retry(
                            best_model_path,
                            {
                                'epoch': current_epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler_lr.state_dict(),
                                'scaler_state_dict': scaler.state_dict(),
                                'val_loss': val_loss,
                                'running_avg_loss': running_avg_loss,
                                'val_losses': val_losses,
                                'running_avg_losses': running_avg_losses,
                                'ema_state': ema.state_dict(),
                                'batch_size': batch_size,
                            }
                        )

                # *Save model checkpoint every N epochs
                if (current_epoch) % args.save_every_epoch == 0:
                    ckpt_path = os.path.join(check_save_dir, f"model{'_test' if Dataset == 'mnist' else ''}_{current_epoch}.pth")
                    save_with_retry(
                        ckpt_path,
                        {
                            "epoch": current_epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scaler_state_dict": scaler.state_dict() if device.type == "cuda" else {},
                            "ema_state": ema.state_dict(),
                            "loss": running_avg_loss,
                            "version": 1,
                        }
                    )

                    #* Log checkpoint at the end of each epoch
                    signature = build_signature(num_channels=num_channels, img_size=img_size, num_classes=num_classes)
                    
                    log_artifact_safe(ckpt_path, artifact_path="model_checkpoints", log_status=LOG_STATUS)
                
                    #* Manage top K models based on running average loss
                    top_models.append((running_avg_loss, ckpt_path))
                    top_models.sort(key=lambda x: x[0])
                    if len(top_models) > args.top_k_models:
                        _, worst_model_path = top_models.pop()
                        if os.path.exists(worst_model_path):
                            os.remove(worst_model_path)

        print(f"\nBest model saved with loss {best_loss:.4f}")
        if best_fid_path:
            print(f"Saved best FID model to {best_fid_path} (FID={best_fid:.4f})")
        # Sampling block (unconditional preview first):
        os.makedirs(f"figures/{args.dataset}/samples/{args.batch_size}", exist_ok=True)
        # Final unconditional preview using EMA weights
        ema.apply_shadow(model)
        try:
            result = sample_images(
                model, noise_scheduler, img_size, device,
                n=16, labels=None, guidance_scale=None, return_intermediates=False
            )
            if isinstance(result, (list, tuple)):
                uncond_preview = result[0]
            else:
                uncond_preview = result
        finally:
            ema.restore(model)
        torchvision.utils.save_image(uncond_preview, f"figures/{args.dataset}/samples/{args.batch_size}/uncond_epoch_{current_epoch}.png", nrow=4, normalize=True)

        # Also save a separate file with EMA weights applied to the model (for easy inference)
        ema_save_path = f"{save_dir}/{model_name}_EMA{'_test' if Dataset == 'mnist' else ''}.pth"
        ema.apply_shadow(model)
        # save checkpoint with ema.state_dict()
        save_with_retry(
            ema_save_path,
            {
                "model_state_dict": model.state_dict(),
                "batch_size": batch_size
            }
        )
        ema.restore(model)
        print(f"EMA model saved as {ema_save_path}")

        # Correct plot_losses call with epoch lists
        plot_losses(
            train_epochs,
            train_epoch_losses,
            val_epoch_points,
            val_losses,
            running_avg_losses,
            save_path=f"figures/{args.dataset}/loss_curves/{args.batch_size}/loss_plot.png"
        )
        plot_fid(fid_scores, fid_epochs, save_path=f"figures/{args.dataset}/fid/{args.batch_size}/fid_plot.png")
        
    except (Exception, KeyboardInterrupt) as e:
        is_kb_interrupt = isinstance(e, KeyboardInterrupt)
        status = "interrupted" if is_kb_interrupt else "error"
        if not is_kb_interrupt:
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_summary = traceback.extract_tb(exc_tb)
            last = tb_summary[-1] if tb_summary else None
            if last:
                print(f"Error: {exc_type.__name__}: {exc_value} at {last.filename}:{last.lineno} in {last.name}")
            # Print full traceback for context
            traceback.print_exc()
        else:
            print("Training interrupted by KeyboardInterrupt")
        _end_mlflow_run(status=status)
        sys.exit(130 if is_kb_interrupt else 1)
    finally:
        _end_mlflow_run(status="finished")


