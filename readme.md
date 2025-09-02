# Text-to-Image

This project will try to create a text-to-image model using the dataset text-to-image-2m from Hugging Face.

The process involves creating a network from scratch using PyTorch, train the model, and then try to generate an image based on a simple text prompt.

## The dataset

The dataset consist of prompts that describes images, and the image itself.

### Goal

The first goal will be to apply noise to the images, then learn a UNET model to predict the noise level an image contain, so it can de-noise it later.
