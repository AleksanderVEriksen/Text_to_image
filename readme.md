# Text-to-Image

This project trains a diffusion (DDPM) UNet to denoise images conditioned on class labels, then generates samples (optionally with classifier‑free guidance) and will later extend toward full text-to-image.

## Dataset

Images (custom WebDataset or MNIST) plus optional textual/label conditioning. MNIST is internally resized to 32×32; custom data can be 64×64.

## Model architecture

![Architecture of the UNET](readme_image_folder/model_diagram_Mermaid.png "Mermaid image of model architecture")

UNet variants (`UNET`, `BasicUNet`) with:

- Sinusoidal time embeddings + MLP fusion.
- Label embeddings with classifier‑free dropout during training.
- Decoder uses upsample + 1×1 conv (instead of transposed conv) to reduce checkerboard artifacts.
- Classifier‑free guidance is applied in the sampler (not inside `forward()`), mixing conditional and unconditional predictions externally.
- AMP support and EMA weights used for validation/sampling to improve sample quality.

## PyTorch with CUDA

Training uses PyTorch, mixed precision (AMP), optional EMA weights, and a DDPM noise scheduler.

## File Structure (Core)

### data.py

Loads and transforms datasets (MNIST resized to 32×32 and scaled to [-1,1]; custom dataset hook). Provides helper accessors.

### model.py

Defines UNET and BasicUNet with sinusoidal time embeddings, label embeddings, and optional classifier‑free guidance (guidance_scale during sampling).

### utils.py

Grouped helper functions: seeding, checkpoints, sampling (forward diffusion reversal), SNR-weighted loss, FID with progress bars, plotting, validation, etc.

New: Inception Score (IS) computation alongside FID.

### train.py

Full training loop: DDPM noise addition, UNet noise prediction, SNR-weighted loss option, periodic validation, FID scheduling, sample grid saving, EMA maintenance, config.json writing.

Validation and sampling use EMA weights. FID and IS can be reported at the same cadence.

### eval.py

Evaluates a saved model: loads weights/config, runs a forward corruption + denoising illustration, generates conditional sample grids, optional classifier‑free guidance.

### generate_from_text.py

Generates images from a label string (“7”, “one two”, “all”), loads config.json to align hyperparameters, supports guidance_scale for sharper conditional results, saves a grid.

### ema.py

Exponential moving average wrapper for stabilizing inference weights.

## Removed

playground (previous exploratory file) is no longer listed.

## Configuration (config.json)

Written once per training run (models/<batch_size>/config.json) recording: version, batch_size, model, embedding_dim, num_classes, max_timesteps (and optionally img_size). eval.py and generate_from_text.py load it to:

- Warn on mismatches (e.g. num_classes).
- Override img_size / max_timesteps if present.
Keep this file with checkpoints for reproducibility.

## How to Run

Activate venv:

```python
./venv/Scripts/Activate
```

### Train

Basic command:

```python
python train.py
```

Key arguments (see train.py parse_args):

- --batch_size (int, default 32): Training batch size (also folder key for weights).
- --epochs (int, default 100): Total training epochs.
- --max_timesteps (int, default 1000): Diffusion steps (DDPM).
- --dataset {mnist|custom} (default mnist): Dataset source. mnist auto-resizes to 32×32.
- --model {UNET|Basic} (default UNET): Architecture choice.
- --num_classes (int, default 10): Label class count.
- --checkpoint (flag): Resume from models/checkpoints/<batch_size>/<model_name>.pth.
- --model_name (str): Base filename for saving (e.g. best_model).
- --val_every (int, default 5): Validation frequency (epochs).
- --val_max_batches (int, default 32): Cap validation batches processed.
- --sample_every_epoch (int, default 50): Generate sample grid on schedule (also at epoch 0).
- --save_every_epoch (int, default 10): Periodic checkpoint saving cadence.
- --augment (flag): Enable data augmentation for MNIST (if implemented in data.py).
- --patience (int, default 5): Early stopping patience on FID stagnation.
- --top_k_models (int, default 3): Keep only lowest-loss recent checkpoints.
- --fid_epoch_calc (int, default 50): Epoch interval to compute FID.
- --is_epoch_calc (int, default 50): Epoch interval to compute Inception Score (IS).
- --seed (int, default 42): Reproducibility.
- --use_weighted_snr (flag): Enable SNR-based weighting of the noise-prediction loss.
- --no_ema_validate (flag): Disable EMA weights during validation (compare raw vs EMA performance).

Implicit:

- img_size chosen by dataset (32 for MNIST, 64 for custom).
- EMA automatically maintained (decay 0.9999).
- config.json saved under models/<batch_size>/.
- Validation runs with EMA weights; metrics logged include train/val losses, FID, and IS.

### Eval

```python
python eval.py --batch_size 32 --model UNET --model_name best_model --test
```

Arguments:

- --batch_size: Must match folder used during training.
- --max_timesteps: Override diffusion steps if not using config.json.
- --test (flag): Use MNIST evaluation path.
- --checkpoint (flag): Load from checkpoints/<batch_size>/<model_name>.pth.
- --EMA (flag): Load EMA weights instead of raw.
- --model {UNET|Basic}
- --model_name (str)
- --num_classes (int)
- --guidance_scale (float): Enable classifier‑free guidance (higher ~ stronger conditioning).
Reads config.json to reconcile num_classes, max_timesteps, img_size.

### Generate From Text / Labels

```python

python generate_from_text.py --label "seven" --model UNET --model_name best_model --batch_size 32 --guidance_scale 3.0
```

Arguments:

- --model_name: Saved weights base file.
- --model {UNET|Basic}
- --label: Single or multi-label string (“7”, “one three”, “2,5,9”, “all”).
- --num_samples (default 16): Grid size.
- --num_classes (default 10)
- --img_size (default 32, overridden by config if present).
- --timesteps (default 1000): Rebuilt scheduler steps (overridden by config).
- --out (default generated.png): Output image file path.
- --batch_size: Folder key for locating weights/config.
- --guidance_scale: Enable classifier‑free guidance (e.g. 0 = none, 1–5 typical range).
Behavior:
- Parses label string into indices.
- Tiles/repeats to match num_samples.
- Loads config.json to adjust hyperparameters.
- Generates final denoised samples from pure noise.

## Diffusion Process

1. Forward: Add Gaussian noise at random timesteps.
2. Model predicts noise ε conditioned on (t, label).
3. Reverse (sampling): Iterate scheduler.timesteps, remove predicted noise.
4. Optional guidance: Run conditional + null labels; blend outputs.

## Guidance (Classifier-Free)

Enabled by:

- Training-time label dropout.
- Inference guidance_scale > 0.
Formula:

```math
ε = ε_uncond + guidance_scale * (ε_cond - ε_uncond)
```

Higher scale sharpens class fidelity but may reduce diversity.

## FID Computation

- Collected every fid_epoch_calc epochs (or adjust to more frequent).
- Real images remapped from [-1,1] to [0,1].
- InceptionV3 avgpool features → Fréchet distance.

## Inception Score (IS)

- Reported alongside FID on generated images only.
- Computed with torchvision InceptionV3 softmax on resized inputs; measures confident class predictions and diversity.
- Higher is better; trends are most useful (absolute values depend on domain and preproc).
- For MNIST, IS is still informative for trends, though the classifier is ImageNet-trained.

## MLflow Tracking

- Autologging enabled (`mlflow.pytorch.autolog(log_models=False)`), tracking URI from `.env` via `MLFLOW_TRACKING_URI`; experiment ID via `MLFLOW_EXPERIMENT_ID`.
- Logs params (device, OS, CUDA info), metrics (`train_loss`, `val_loss`, `fid_score`, `is_score`), and artifacts (checkpoints, sample grids).
- System metrics logging enabled and sampled periodically.
- To disable logging if needed, `utils.disable_mlflow_logging()` guards calls and ends the active run.

## EMA

Maintains smoothed parameter copies for better sample quality:

- Apply for inference (ema.apply_shadow).
- Restore afterward (ema.restore) if needed.

## Goals (Incremental)

1. Noise prediction baseline (done).
2. Time embedding integration (done).
3. Conditional label guidance (done).
4. Classifier-free guidance (done).
5. Larger dataset + textual encoder (future).
6. VAE + tokenizer for full prompt conditioning (planned).

## Run Tips

- Lower fid_epoch_calc for faster feedback (e.g., 10 or equal to val_every).
- Use guidance_scale 2–4 for sharper digits.
- Maintain consistent batch_size folder across train/eval/generate.

## Example

Generate digit ‘7’ with guidance:

```python
python generate_from_text.py --label 7 --model UNET --model_name best_model --batch_size 32 --guidance_scale 3
```

## Troubleshooting

- Shape mismatch (28 vs 16): Ensure MNIST resized to 32×32 or re-enable output interpolation.
- FID always None: Reduce --fid_epoch_calc.
- Weights not found: Check models/<batch_size>/ directory and model_name correctness.
 - Weights not found: Use the fetch script below to download and place `.pth` files.
- Guidance ineffective: Confirm label dropout active during training; ensure guidance_scale passed at generation.

## Tests

|*Basic UNET - No added timesteps*|
|:--:|
|![MNIST dataset test on Basic UNET -- Did not predict well due to no added timestep](readme_image_folder/image.png "Basic UNET - No added timesteps")|

|*Basic UNET - Added timesteps*|
|:--:|
|![MNIST dataset test on Basic UNET -- Added Timestep to UNET](readme_image_folder/image-1.png "Added timesteps")|

|*Prediction with Custom dataset*|
|:--:|
|![Prediction with custom dataset](readme_image_folder/image-2.png "Custom dataset")|

|*Evaluation of trained UNET on MNIST dataset - 10 timesteps*|
|:--:|
|![Evaluation of trained UNET on MNIST dataset](readme_image_folder/image-4.png "Eval on model trained on MNIST")|

|*Evaluation of trained UNET on MNIST dataset - 1000 timesteps*|
|:--:|
|![Evaluation of trained UNET on MNIST dataset](readme_image_folder/image-5.png "Eval on model trained on MNIST")|

### Test with attached labels

|*Evaluation of trained UNET on MNIST dataset - 1000 timesteps*|
|:--:|
|![Evaluation of trained UNET on MNIST dataset](readme_image_folder/image-6.png "Eval on model trained on MNIST")|

|*Generated images of 7 from text*|
|:--:|
|![Generated images of 7 from text prompt](readme_image_folder/image-7.png "Reconstructed images of 7")|

## Tests with added projections - 20 epochs

|* Test with added projections, and forced 32x32 image|
|:--:|
|![Generated images of 7 from text prompt](readme_image_folder/image-8.png "Reconstructed images of 7")|

|![Eval](readme_image_folder/image-9.png "Eval")|
|:--:|

|![Generated samples denoised](readme_image_folder/image-10.png)|
|:--:|

|![Generated samples](readme_image_folder/image-11.png)|
|:--:|

## Model Weights

This repository does not store `.pth` weight files in git history or Git LFS by default. Training outputs remain local and ignored. When you want to publish a finalized checkpoint, place it under `models/published/` and commit it.

Download steps:

- Add direct URLs to scripts/model_urls.json for the files you need (e.g., GitHub Releases or Hugging Face raw file links).
- Run the downloader from repo root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/fetch_models.ps1
# Re-download if files already exist
powershell -ExecutionPolicy Bypass -File scripts/fetch_models.ps1 -Force
```

Notes:

- `models/**/*.pth` is ignored by git to prevent accidental commits; `models/published/*.pth` is whitelisted to allow intentional publishing.
- If you prefer hosting weights externally, you can still use the downloader above.
- CI can populate `scripts/model_urls.json` via secrets before running.

### Publish finalized checkpoints

To publish a specific weight file without using the published folder, you can force-add it:

```powershell
git add -f models/mnist/64/best_model.pth
git commit -m "Publish finalized checkpoint: best_model.pth"
git push
```

Alternatively, save finalized weights in [models/published](models/published) and commit normally.

## Tests with added projections - 300 epochs

|* Test with added projections, and forced 32x32 image|
|:--:|
|![Generated images of 7 from text prompt](readme_image_folder/image-12.png "Reconstructed images of 7")|
|:--:|

|![Eval](readme_image_folder/image-14.png "Eval")|
|:--:|

|![Generated samples denoised](readme_image_folder/image-13.png)|
|:--:|
