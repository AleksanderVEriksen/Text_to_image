import matplotlib.pyplot as plt
import mlflow
from torchvision import transforms
import numpy as np, os, torch, torchvision, scipy.linalg, PIL.Image
from torchvision.models import inception_v3, Inception_V3_Weights
import torch.nn.functional as F
from tqdm.auto import tqdm

from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset as TorchDataset

from typing import Any, cast, Optional, overload, Tuple, List, Literal, Union
from data import get_dataset, get_mnist_dataset
# =========================================================
# utils.py
# =========================================================

# Global logging status guard (default True; train.py may override)
LOG_STATUS: bool = True

# Defines the transformation to convert images to tensors normalized to [-1, 1]
transform = transforms.Compose([
    transforms.Resize((32, 32)),    # Added: ensure 32x32 so 2^5 downsamples align
    transforms.ToTensor(),          # [0,1]
    transforms.Lambda(lambda x: 2 * x - 1)  # [-1,1]
])

# =========================================================
# utils.py
# Grouped utility functions with clarifying comments.
# Sections:
# 1. Seeding / Reproducibility
# 2. File & Checkpoint Helpers
# 3. Data / Dataloader Helpers
# 4. Sampling & Snapshot Functions
# 5. Loss / SNR Weighting
# 6. Metrics (FID)
# 7. Plotting
# 8. Misc / Validation Loop
# =========================================================

# ---------------------------------------------------------
# 1. Seeding / Reproducibility
# ---------------------------------------------------------
def set_global_seed(seed: int):
    """Set seeds for Python, NumPy, Torch (CPU/CUDA) for reproducibility."""
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------
# 2. File & Checkpoint Helpers
# ---------------------------------------------------------
def save_with_retry(path, obj, retries=3):
    """Save torch object with simple retry mechanism (handles transient IO errors)."""
    import torch, time, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            torch.save(obj, path)
            return True
        except Exception as e:
            if attempt == retries:
                print(f"Failed saving {path}: {e}")
                return False
            time.sleep(0.5 * attempt)

def save_config(config_dict, path):
    """Persist a JSON config dictionary to disk."""
    import json, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)

def load_model_weights(model, model_name, batch_size, device, use_checkpoint=False, use_ema=False, strict=True):
    """Load model weights from possible candidate paths (EMA / checkpoint / base)."""
    import os, torch
    candidates = []
    if use_ema:
        candidates.append(f"models/{batch_size}/{model_name}_EMA_test.pth")
    if use_checkpoint:
        candidates.append(f"models/checkpoints/{batch_size}/{model_name}.pth")
    candidates.append(f"models/{batch_size}/{model_name}.pth")
    for p in candidates:
        if not os.path.isfile(p):
            continue
        try:
            ckpt = torch.load(p, map_location=device)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state, strict=strict)
            print(f"Loaded weights: {p}")
            return ckpt if isinstance(ckpt, dict) else {}
        except Exception as e:
            print(f"Failed {p}: {e}")
    print("No weights loaded.")
    return None

# ---------------------------------------------------------
# 3. Data / Dataloader Helpers
# ---------------------------------------------------------
def collate_fn(batch):
    images = []
    labels = []
    for sample in batch:
        if isinstance(sample, dict):
            images.append(sample.get('image', sample.get('jpg', None)))
            labels.append(sample.get('label', -1))
        else:
            # Handle tuple case (image, label)
            if isinstance(sample, tuple) and len(sample) >= 1:
                img = sample[0]
                label = sample[1] if len(sample) > 1 else -1
                if isinstance(img, PIL.Image.Image):
                    img = transform(img)
                images.append(img)
                labels.append(label)
            else:
                images.append(sample)
                labels.append(-1)

    # Stack images and ensure they're tensors
    images = [img if isinstance(img, torch.Tensor) else transform(img) for img in images]
    images = torch.stack(images, dim=0)
    
    # Convert labels to tensor
    labels = torch.tensor([int(l) if not isinstance(l, torch.Tensor) else int(l.item()) 
                        for l in labels], dtype=torch.long)
    return images, labels

def estimate_dataset_stats(dataloader, max_batches=20, device='cpu'):
    """Estimate mean and variance over a limited number of batches."""
    import torch
    cnt = 0
    mean = 0
    M2 = 0
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            imgs = imgs.to(device).float()
            batch_mean = imgs.mean()
            batch_var = imgs.var()
            delta = batch_mean - mean
            cnt += 1
            mean += delta / cnt
            M2 += batch_var
            if cnt >= max_batches:
                break
    return {"approx_mean": float(mean), "approx_var": float(M2 / max(1, cnt))}

# ---------------------------------------------------------
# 4. Sampling & Snapshot Functions
# ---------------------------------------------------------
@overload
def sample_images(
    model, 
    scheduler, 
    img_size: int, 
    device, 
    n: int = 16,
    labels: Optional[Union[int, Tensor]] = None,
    return_intermediates: Literal[True] = True,
    guidance_scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor, List[Tensor]]: ...
@overload
def sample_images(
    model, 
    scheduler, 
    img_size: int, 
    device, 
    n: int = 16,
    labels: Optional[Union[int, Tensor]] = None,
    return_intermediates: Literal[False] = False,
    guidance_scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor]: ...

def sample_images(
    model,
    scheduler,
    img_size: int,
    device,
    n: int = 16,
    labels: Optional[Union[int, Tensor]] = None,
    return_intermediates: bool = False,
    guidance_scale: Optional[float] = None,
):
    """Generate n final denoised samples (optionally capture intermediates).
    Classifier-free guidance is applied here (external to model) if guidance_scale provided.
    """
    model.eval()
    with torch.no_grad():
        x = torch.randn(n, model.in_channels, img_size, img_size, device=device)
        if labels is not None:
            if isinstance(labels, int):
                labels = torch.full((n,), labels, dtype=torch.long, device=device)
            elif isinstance(labels, torch.Tensor):
                labels = labels.to(device).view(-1).long()
                if labels.shape[0] != n:
                    labels = labels.repeat(n)[:n]
        intermediates = []
        for t in scheduler.timesteps:
            t_scalar = int(t.item()) if hasattr(t, "item") else int(t)
            t_batch = torch.tensor([t_scalar] * n, device=device)
            if guidance_scale is not None and labels is not None:
                eps_cond = model(x, t_batch, labels=labels)
                eps_uncond = model(x, t_batch, labels=None)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = model(x, t_batch, labels=labels)
            step_out = scheduler.step(eps, t, x)
            x = step_out.prev_sample
            if return_intermediates:
                intermediates.append(x.detach().clone())
        x0 = x.detach()
        minv, maxv = float(x0.min()), float(x0.max())
        if minv >= -1.2 and maxv <= 1.2:
            vis = ((x0 + 1) / 2).clamp(0, 1)
        elif 0.0 <= minv and maxv <= 1.0:
            vis = x0.clamp(0, 1)
        elif maxv <= 255.0:
            vis = (x0 / 255.0).clamp(0, 1)
        else:
            vis = (x0 - minv) / (maxv - minv + 1e-8)
            vis = vis.clamp(0, 1)
        ts_used = scheduler.timesteps.clone().detach()
        if return_intermediates:
            return vis, ts_used, intermediates
        return vis, ts_used

def sample_snapshots_by_t(model, scheduler, img_size, device, *, in_channels=1,
                        labels=None, timesteps_subset=None, seed=None):
    """Capture intermediate x_t states for a single sample at specified timesteps."""
    import torch
    model.eval()
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    x = torch.randn((1, in_channels, img_size, img_size), device=device, generator=gen)
    snapshots, taken_ts = [], []
    if labels is not None:
        if isinstance(labels, int):
            labels = torch.tensor([labels], dtype=torch.long, device=device)
        elif isinstance(labels, torch.Tensor):
            labels = labels.to(device).view(1).long()
        else:
            raise TypeError("labels must be int or tensor for snapshots.")
    wanted = set(int(t) for t in (timesteps_subset or []))
    with torch.no_grad():
        for t in scheduler.timesteps:
            t_int = int(t.item()) if hasattr(t, "item") else int(t)
            eps = model(x, torch.tensor([t_int], device=device), labels)
            step_out = scheduler.step(eps, t, x)
            x = step_out.prev_sample
            if t_int in wanted:
                snapshots.append(x.detach().clone())
                taken_ts.append(t_int)
    if not snapshots:
        return torch.empty(0, in_channels, img_size, img_size, device=device), torch.tensor([], dtype=torch.long)
    return torch.cat(snapshots, dim=0), torch.tensor(taken_ts, dtype=torch.long, device=device)

def sample_intermediates(model, scheduler, img_size, device, n=1, in_channels=1, labels=None, capture_ts=None):
    """Return (snapshots, captured_timesteps) for one sample along the denoising path."""
    model.eval()
    x = torch.randn((n, in_channels, img_size, img_size), device=device)
    out_imgs = []
    out_ts = []
    wanted = set(int(t) for t in (capture_ts or []))
    with torch.no_grad():
        for t in scheduler.timesteps:
            t_int = int(t.item())
            eps = model(x, torch.tensor([t_int], device=device), labels)
            step = scheduler.step(eps, t, x)
            x = step.prev_sample
            if t_int in wanted:
                out_imgs.append(x.detach().clone())
                out_ts.append(t_int)
    if not out_imgs:
        return torch.empty(0, in_channels, img_size, img_size), torch.tensor([], dtype=torch.long)
    return torch.cat(out_imgs, dim=0), torch.tensor(out_ts, dtype=torch.long, device=device)
# ---------------------------------------------------------
# 5. Loss / SNR Weighting
# ---------------------------------------------------------
def compute_snr(alphas_cumprod, timesteps):
    """Compute SNR = alpha_cumprod / (1 - alpha_cumprod) for given timesteps."""
    a = alphas_cumprod[timesteps]
    return a / (1 - a)

def weighted_noise_loss(eps_pred, eps_target, alphas_cumprod, timesteps, min_snr_gamma=5.0):
    """Apply SNR-based weighting to noise prediction MSE to balance timestep difficulty."""
    import torch
    snr = compute_snr(alphas_cumprod, timesteps)
    w = (snr.clamp(max=min_snr_gamma) / snr)
    loss = (w * (eps_pred - eps_target).pow(2).mean(dim=(1, 2, 3))).mean()
    return loss

# ---------------------------------------------------------
# 6. Metrics (FID)
# ---------------------------------------------------------
def calculate_fid(real_images, generated_images, device='cuda', show_progress=False, chunk_size=64):
    """Compute Fréchet Inception Distance between real and generated batches.
    If show_progress=True, display a tqdm bar over feature extraction.
    Args:
        real_images (Tensor): (N,C,H,W) in [0,1]
        generated_images (Tensor): (N,C,H,W) in [0,1]
        device: torch device
        show_progress (bool): enable tqdm
        chunk_size (int): batch size for feature extraction
    """
    # Expect both in [0,1]
    assert real_images.min() >= -0.01 and real_images.max() <= 1.01
    assert generated_images.min() >= -0.01 and generated_images.max() <= 1.01

    inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                            transform_input=False).to(device)
    inception.eval()

    def prep(imgs):
        imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        imgs = torch.clamp(imgs, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        return (imgs - mean) / std

    def collect_features(imgs):
        feats = []
        handle = inception.avgpool.register_forward_hook(lambda m,i,o: feats.append(o.flatten(1)))
        iterator = range(0, imgs.shape[0], chunk_size)
        if show_progress:
            iterator = tqdm(iterator, desc="FID features", leave=False)
        with torch.no_grad():
            for start in iterator:
                batch = imgs[start:start+chunk_size].to(device)
                inception(prep(batch))
        handle.remove()
        return torch.cat(feats, dim=0).cpu().numpy()

    real_feats = collect_features(real_images)
    gen_feats  = collect_features(generated_images)

    mu_r, mu_g = real_feats.mean(0), gen_feats.mean(0)
    sigma_r = np.cov(real_feats, rowvar=False)
    sigma_g = np.cov(gen_feats, rowvar=False)
    diff = mu_r - mu_g
    covmean = scipy.linalg.sqrtm(sigma_r.dot(sigma_g))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(fid)

def calculate_inception_score(images, device='cuda', show_progress=False, chunk_size=64):
    """Compute Inception Score (IS) on generated images in [0,1].
    IS = exp(E_x KL(p(y|x) || p(y))). Uses Inception v3 softmax.
    """
    assert images.min() >= -0.01 and images.max() <= 1.01
    inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                            transform_input=False).to(device)
    inception.eval()

    def prep(imgs):
        imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        imgs = torch.clamp(imgs, 0, 1)
        return imgs

    preds = []
    iterator = range(0, images.shape[0], chunk_size)
    if show_progress:
        iterator = tqdm(iterator, desc="IS features", leave=False)
    with torch.no_grad():
        for start in iterator:
            batch = prep(images[start:start+chunk_size].to(device))
            logits = inception(batch)
            prob = torch.softmax(logits, dim=1)
            preds.append(prob)
    probs = torch.cat(preds, dim=0)
    py = probs.mean(dim=0, keepdim=True)
    kl = (probs * (probs.log() - py.log())).sum(dim=1)
    return float(torch.exp(kl.mean()).item())

# ---------------------------------------------------------
# 7. Plotting
# ---------------------------------------------------------
def plot_losses(train_epochs, train_losses, val_epochs, val_losses, running_avg_losses, save_path="figures/loss_plot.png"):
    """Plot training & validation loss curves along with running average."""
    import os, matplotlib.pyplot as plt
    if not train_losses and not val_losses:
        print("No loss data to plot.")
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    if train_losses:
        plt.plot(train_epochs, train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_epochs, val_losses, label='Val Loss')
    if running_avg_losses:
        plt.plot(val_epochs, running_avg_losses, label='Running Avg Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {save_path}")

def plot_fid(fids, fid_epochs, save_path="figures/fid_plot.png", save_json=True):
    """Plot FID values over epochs with color-coded segments; optionally save JSON."""
    import os, json, matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    if not fids:
        print("No FID data.")
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    for i in range(len(fids)):
        color = ('green' if fids[i] < 5 else
                'yellow' if fids[i] < 20 else
                'orange' if fids[i] < 50 else 'red')
        if i == 0:
            plt.plot([fid_epochs[i]], [fids[i]], marker='o', color=color)
        else:
            plt.plot([fid_epochs[i - 1], fid_epochs[i]],
                    [fids[i - 1], fids[i]], marker='o', color=color)
    plt.xlabel("Epoch")
    plt.ylabel("FID")
    plt.title("FID over epochs")
    plt.grid(alpha=0.3)
    legend = [
        Patch(facecolor='green', label='<5 Excellent'),
        Patch(facecolor='yellow', label='5–20 Good'),
        Patch(facecolor='orange', label='20–50 Acceptable'),
        Patch(facecolor='red', label='>=50 Poor'),
    ]
    plt.legend(handles=legend)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"FID plot saved to {save_path}")
    if save_json:
        with open(os.path.splitext(save_path)[0] + ".json", "w") as f:
            json.dump({"epochs": fid_epochs, "fids": fids}, f, indent=2)

# ---------------------------------------------------------
# 8. Misc / Validation Loop
# ---------------------------------------------------------
def validate(model, epochs, val_dataloader, noise_scheduler, loss_fn, device,
            max_batches=None, calculate_fid_score=False, fid_epoch_calc=10,
            calculate_is_score=False, is_epoch_calc=10,
            img_size=32, show_progress=True, fid_progress=True, fid_min_samples=512):
    """Validation loop computing average loss and optional FID with progress bars."""
    import torch
    model.eval()
    val_loss = 0.0
    batches = 0
    fid_score = None
    is_score = None
    images_accum = []
    labels_ref = None

    loader_iter = val_dataloader
    if show_progress:
        loader_iter = tqdm(val_dataloader, desc="Validate", leave=False, 
                        position=0, dynamic_ncols=True)

    with torch.no_grad():
        for batch in loader_iter:
            if isinstance(batch, (list, tuple)):
                images = batch[0]; labels = batch[1] if len(batch) > 1 else None
            else:
                images = batch; labels = None
            images = images.to(device)
            if labels is not None:
                labels = labels.to(device).long()
                labels_ref = labels
            total_steps = noise_scheduler.config.num_train_timesteps
            t = torch.randint(0, total_steps, (images.size(0),), device=device).long()
            noise = torch.randn_like(images)
            noisy = noise_scheduler.add_noise(images, noise, t)
            pred = model(noisy, t, labels=labels)
            val_loss += loss_fn(pred, noise).item()
            batches += 1
            if calculate_fid_score:
                images_accum.append(images.detach().cpu())
            if max_batches and batches >= max_batches:
                break
    avg_val_loss = val_loss / max(1, batches)
    do_fid = calculate_fid_score and (epochs % fid_epoch_calc == 0)
    do_is = calculate_is_score and (epochs % is_epoch_calc == 0)
    if do_fid and images_accum:
        real_batch = torch.cat(images_accum, dim=0)
        if real_batch.size(0) < fid_min_samples:
            # Accumulate more by reusing loader until threshold (optional)
            pass
        real_batch = (real_batch + 1) / 2  # [-1,1] -> [0,1]
        gen_batch, _ = sample_images(
            model, noise_scheduler, img_size, device,
            n=real_batch.shape[0], labels=labels_ref if labels_ref is not None else None,
            guidance_scale=None,  # unconditional for FID stability
            return_intermediates=False
        )
        fid_score = calculate_fid(real_batch.to(device), gen_batch.to(device), device=device, show_progress=fid_progress)
        if do_is:
            is_score = calculate_inception_score(gen_batch.to(device), device=device, show_progress=fid_progress)
        else:
            is_score = None
    return {
        "val_loss": avg_val_loss,
        "fid_score": fid_score,
        "is_score": is_score
    }

def text_to_label(label, max_num_classes: int = 10):
    """Convert text/numeric label to tensor index."""
    if isinstance(label, int):
        return label
    if isinstance(label, str):
        try:
            # Try direct numeric conversion first
            return int(label)
        except ValueError:
            # Map text to numbers
            if max_num_classes > 10:
                raise ValueError("Text labels not supported for num_classes > 10")
            label_map = {
                'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
                'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
            }
            label = label.lower()
            if label in label_map:
                return label_map[label]
    raise ValueError(f"Unsupported label format: {label}")

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

def normalize_per_sample(tensor, min_val=-1, max_val=1):
    """Normalize each sample in a batch independently to a given range.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (N,C,H,W)
        min_val (float): Target minimum value
        max_val (float): Target maximum value
    
    Returns:
        torch.Tensor: Normalized tensor
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")
        
    # Get dimensions
    B, C, H, W = tensor.shape
    
    # Reshape to (B,C*H*W) for per-sample normalization
    flat = tensor.view(B, -1)
    
    # Get min/max per sample
    min_per_sample = flat.min(dim=1, keepdim=True)[0]
    max_per_sample = flat.max(dim=1, keepdim=True)[0]
    
    # Normalize
    scale = (max_val - min_val) / (max_per_sample - min_per_sample + 1e-8)
    normalized = (flat - min_per_sample) * scale + min_val
    
    # Reshape back
    return normalized.view(B, C, H, W)

def timesteps_to_str(ts, max_items=20):
    lst = list(ts.tolist())
    if len(lst) > max_items:
        return ", ".join(map(str, lst[:max_items])) + ", ..."
    return ", ".join(map(str, lst))

# Load image to tensor
def load_single_img_to_tensor(dataset):
    sample = next(iter(dataset))
    image = sample['jpg']
    image = transform(image)  # (C, H, W), normalisert til [0, 1]
    return image

# Load dataset of images to tensor
def sample_to_tensor(sample):
    if isinstance(sample, dict):
        return transform(sample['jpg'])
    else:
        return transform(sample[0])
# ==================== Load data from data.py ===============================

def load_data_from_dataset(dataset_name: str, batch_size: int, Augment: bool):
        # *Load dataset from data.py
    if dataset_name == "custom":
        print("Training on custom dataset")

        train = get_dataset(train=True)
        test = get_dataset(test=True)
        val = get_dataset(val=True)

        train_ds = cast(TorchDataset[Any], train)
        val_ds = cast(TorchDataset[Any], val)
        test_ds = cast(TorchDataset[Any], test)

        train_dataloader = cast(DataLoader[Any], DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn))
        val_dataloader = cast(DataLoader[Any], DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn))
        test_dataloader = cast(DataLoader[Any], DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn))

        return train_dataloader, val_dataloader, test_dataloader
    
    elif dataset_name == "mnist":
        # *Load example dataset for testing
        print("\n---Training on MNIST dataset---")
        train_ = get_mnist_dataset(train=True, augment=Augment)
        test_ = get_mnist_dataset(train=False, augment=False)

        train_size = int(len(train_) * 0.8)
        val_size = len(train_) - train_size
        train, val = random_split(train_, [train_size, val_size] )

        train_dataloader = DataLoader(train, batch_size=batch_size, collate_fn=collate_fn)
        val_dataloader = DataLoader(val, batch_size=batch_size, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_, batch_size=batch_size, collate_fn=collate_fn)
        
        return train_dataloader, val_dataloader, test_dataloader
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
# =========================================================
from typing import Any, Union, List, cast
from mlflow.types import ColSpec, TensorSpec
from mlflow.models import ModelSignature
from mlflow.types import Schema

def build_signature(num_channels: int, img_size: int, num_classes: int) -> ModelSignature:
    input_specs: List[Union[TensorSpec, ColSpec]] = [
        TensorSpec(np.dtype(np.float32), (-1, num_channels, img_size, img_size), name="x"),
        TensorSpec(np.dtype(np.int64),   (-1,), name="timesteps"),
    ]
    if num_classes:
        input_specs.append(TensorSpec(np.dtype(np.int64), (-1,), name="labels"))
    output_specs: List[Union[TensorSpec, ColSpec]] = [
        TensorSpec(np.dtype(np.float32), (-1, num_channels, img_size, img_size), name="pred_noise"),
    ]
    # Cast to satisfy Schema’s expected parameter type
    return ModelSignature(
        inputs=Schema(cast(List[Union[TensorSpec, ColSpec]], input_specs)),
        outputs=Schema(cast(List[Union[TensorSpec, ColSpec]], output_specs)),
    )
# ==================== ML FLOW ===============================

def disable_mlflow_logging() -> None:
    # Disable MLflow and end the active run once (avoid train import to prevent circular deps)
    global LOG_STATUS
    LOG_STATUS = False
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run(status="error")
    except Exception:
        # ignore end_run errors during shutdown
        pass
    try:
        mlflow.pytorch.autolog(disable=True)
    except Exception:
        pass

def log_metrics_safe(metrics: dict, step: int, log_status: Optional[bool] = None) -> None:
    # Unified metrics logging with BAD_REQUEST handling
    if log_status is None:
        log_status = LOG_STATUS
    if not log_status:
        return
    try:
        mlflow.log_metrics(metrics, step=step)
    except Exception as e:
        print(f"MLflow metrics logging failed: {e}")
        if "BAD_REQUEST" in str(e):
            disable_mlflow_logging()

def log_artifact_safe(local_path: str, artifact_path: Optional[str] = None, log_status: Optional[bool] = None) -> None:
    # Optional: reuse for artifact logging
    if log_status is None:
        log_status = LOG_STATUS
    if not log_status:
        return
    try:
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
    except Exception as e:
        print(f"MLflow artifact logging failed: {e}")
        if "BAD_REQUEST" in str(e):
            disable_mlflow_logging()

def log_model_safe(model, name: str, signature, pip_requirements=None, metadata=None, log_status: Optional[bool] = None):
    # Optional: reuse if you keep mlflow.pytorch.log_model
    if log_status is None:
        log_status = LOG_STATUS
    if not log_status:
        return None
    try:
        return mlflow.pytorch.log_model(
            model, name=name, signature=signature,
            pip_requirements=pip_requirements, metadata=metadata
        )
    except Exception as e:
        print(f"MLflow model logging failed: {e}")
        if "BAD_REQUEST" in str(e):
            disable_mlflow_logging()
        return None