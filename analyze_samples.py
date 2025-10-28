import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import csv

samples_dir = r"d:\workspace\Text_to_image\figures\samples"
out_dir = os.path.join(samples_dir, "analysis")
os.makedirs(out_dir, exist_ok=True)

# collect png files
files = [os.path.join(samples_dir, f) for f in os.listdir(samples_dir) if f.lower().endswith(".png")]
if not files:
    print("No PNG files found in", samples_dir)
    sys.exit(1)

summary = []
imgs = []
names = []
for path in sorted(files):
    img = Image.open(path).convert("L")  # grayscale
    arr = np.asarray(img).astype(np.float32) / 255.0
    imgs.append(arr)
    names.append(os.path.basename(path))
    mn, mx, mean, std = float(arr.min()), float(arr.max()), float(arr.mean()), float(arr.std())
    median = float(np.median(arr))
    # contrast metric (std relative to mean)
    contrast = std / (mean + 1e-8)
    summary.append({
        "file": os.path.basename(path),
        "shape": arr.shape,
        "min": mn,
        "max": mx,
        "mean": mean,
        "median": median,
        "std": std,
        "contrast": contrast
    })

# write CSV summary
csv_path = os.path.join(out_dir, "samples_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)
print("Wrote summary to", csv_path)

imgs = np.stack(imgs, axis=0)  # (N,H,W)
N, H, W = imgs.shape
print(f"Loaded {N} images, shape each: {H}x{W}")

# global stats
global_min = imgs.min()
global_max = imgs.max()
global_mean = imgs.mean()
global_std = imgs.std()
print("global min/max/mean/std:", global_min, global_max, global_mean, global_std)

# per-pixel std across samples (shows where pixels vary)
pixel_std = imgs.std(axis=0)  # HxW
plt.figure(figsize=(4,4))
plt.title("Per-pixel std across samples")
plt.axis("off")
plt.imshow(pixel_std, cmap="magma")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "per_pixel_std.png"), dpi=150)
plt.close()

# save histogram of pixel values aggregated across all images
plt.figure(figsize=(5,3))
plt.hist(imgs.ravel(), bins=100, color="gray")
plt.title("Pixel value histogram (all samples)")
plt.xlabel("pixel value (0-1)")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "pixel_histogram.png"), dpi=150)
plt.close()

# montage of few highest-variance images and lowest-variance
stds = np.array([s["std"] for s in summary])
order = np.argsort(stds)
low_idx = order[:min(9, N)]
high_idx = order[-min(9, N):][::-1]

def save_montage(indices, fname):
    k = len(indices)
    cols = min(4, k)
    rows = (k + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axs = np.atleast_2d(axs)
    for i, idx in enumerate(indices):
        r = i // cols
        c = i % cols
        axs[r,c].imshow(imgs[idx], cmap="gray", vmin=0, vmax=1)
        axs[r,c].axis("off")
        axs[r,c].set_title(f"{names[idx]}\nstd={stds[idx]:.4f}")
    # hide unused axes
    for i in range(k, rows*cols):
        r = i // cols
        c = i % cols
        axs[r,c].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()

save_montage(low_idx, "lowest_variance_samples.png")
save_montage(high_idx, "highest_variance_samples.png")

# simple rules and warnings
warns = []
if global_std < 0.01:
    warns.append("Very low global std -> generated images are nearly constant (collapse).")
if 0.45 <= global_mean <= 0.55:
    warns.append("Global mean near 0.5 -> images may be mid-gray (mapping issue: double-mapped or normalized incorrectly).")
if len(warns) == 0:
    warns.append("No immediate collapse/mid-gray warning from global stats.")
print("Warnings:")
for w in warns:
    print(" -", w)

# write a short text report
report = os.path.join(out_dir, "report.txt")
with open(report, "w") as f:
    f.write("Summary of samples\n\n")
    for s in summary:
        f.write(f"{s['file']}: min={s['min']:.4f} max={s['max']:.4f} mean={s['mean']:.4f} std={s['std']:.6f} contrast={s['contrast']:.6f}\n")
    f.write("\nGlobal min/max/mean/std: {:.6f}/{:.6f}/{:.6f}/{:.6f}\n\n".format(global_min, global_max, global_mean, global_std))
    f.write("Warnings:\n")
    for w in warns:
        f.write(" - " + w + "\n")
print("Report saved to", report)
print("Analysis images saved to", out_dir)