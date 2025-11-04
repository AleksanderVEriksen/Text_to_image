# Text-to-Image

This project will try to create a text-to-image model using the dataset text-to-image-2m from Hugging Face.

The process involves creating a network from scratch using PyTorch, train the model, and then try to generate an image based on a simple text prompt.

## The dataset

The dataset consist of prompts that describes images, and the image itself.

## PyTorch with Cuda enabled

PyTorch will be utilized to create a UNET model as well as training the model. It will also be used to enable training on a GPU, which will make the training time faster.

## File structure

The files are divided into tasks that is needed to be done in order to train the model. The files are data, model, utils, playground, train and eval.

- The file Data prepares the data by fetching it from HuggingFace, and then loading it for usage.
- Model is used to create the UNET model
- utils are several helper functions used in different files which prevent code redundancy
- playground is used to inspect and analyze the dataset, wit h the help of helper functions explained previously
- train is used to train the model with the noisy images from the dataset
- eval is used to evaluate the trained/saved models

## How to run

Commands to use to run this program. Args for train and eval are not necessary, but should be used if one want to train or eval on custom configurations. Only **epochs** for *train.py* should be considered at all times due to how it affect the learning of the model.

### Go into venv

``./venv/Scripts/Activate``

### Run train

`Commands: python train.py`

- `Args:`
  - `--batch_size` (default: 32) - Training batch size
  - `--epochs` (default: 200) - Number of training epochs
  - `--max_timesteps` (default: 1000) - Number of diffusion timesteps
  - `--num_classes` (default: 10) - Number of label classes for conditioning
  - `--img_size` (default: 28) - Image size (assumes square)
  - `--model` (default: "UNET", choices: ['UNET', 'Basic']) - Model architecture to use
  - `--model_name` (default: "model") - Name for saving/loading model
  - `--test` (flag) - Use MNIST dataset for testing
  - `--checkpoint` (flag) - Resume from checkpoint
  - `--save_every` (default: 5) - Save checkpoint every N epochs
  - `--sample_every` (default: 2) - Generate samples every N epochs
  - `--lr` (default: 5e-5) - Initial learning rate
  - `--weight_decay` (default: 0.01) - Weight decay for optimizer

### Run eval

`Commands: python eval.py`

- `Args:`
  - `--batch_size` (default: 32) - Batch size for evaluation
  - `--max_timesteps` (default: 1000) - Number of diffusion timesteps
  - `--num_classes` (default: 10) - Number of label classes for conditioning
  - `--img_size` (default: 28) - Image size (assumes square)
  - `--model` (default: "UNET", choices: ['UNET', 'Basic']) - Model architecture to use
  - `--model_name` (default: "model") - Name of model to load
  - `--test` (flag) - Use MNIST dataset for testing
  - `--checkpoint` (flag) - Load from checkpoint
  - `--EMA` (flag) - Use EMA weights for evaluation
  - `--num_samples` (default: 16) - Number of samples to generate
  - `--save_path` (default: "figures") - Path to save evaluation results

## Goal

### Required

    - Requires VAE (compress input, then decode it back)
    - Tokenizer and encoder (text string into numerical representation)
    - UNET (numeric representation can be feed into UNET as conditioning )
    - Noise Scheduler (PNDMScheduler, LMSDiscreteScheduler)

#### Step 1

The first goal will be to apply noise to the images, then learn a UNET model to predict the noise level an image contain, so it can de-noise it later.

#### Step 2

Create a UNET that takes in "timesteps" in order to predict the denoise level of an image.

#### Step 3

Adjust noise level for proper predictions

#### Step 4

Add conditional prediction to UNET by adding labels.

#### Step 5

Add sampler to reconstruct images from noise to test the model

## Tests

|*Basic UNET - No added timesteps*|
|:--:|
|![MNIST dataset test on Basic UNET -- Did not predict well due to no added timestep](image.png "Basic UNET - No added timesteps")|

|*Basic UNET - Added timesteps*|
|:--:|
|![MNIST dataset test on Basic UNET -- Added Timestep to UNET](image-1.png "Added timesteps")|

|*Prediction with Custom dataset*|
|:--:|
|![Prediction with custom dataset](image-2.png "Custom dataset")|

|*Evaluation of trained UNET on MNIST dataset - 10 timesteps*|
|:--:|
|![Evaluation of trained UNET on MNIST dataset](image-4.png "Eval on model trained on MNIST")|
|:--:|

|*Evaluation of trained UNET on MNIST dataset - 1000 timesteps*|
|:--:|
|![Evaluation of trained UNET on MNIST dataset](image-5.png "Eval on model trained on MNIST")|
|:--:|

### Test with attached labels

|*Evaluation of trained UNET on MNIST dataset - 1000 timesteps*|
|:--:|
|![Evaluation of trained UNET on MNIST dataset](image-6.png "Eval on model trained on MNIST")|
|:--:|

|*Generated images of 7 from text*|
|:--:|
|![Generated images of 7 from text prompt](image-7.png "Reconstructed images of 7")|
|:--:|
