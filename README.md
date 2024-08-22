# Stable Diffusion XL Image Batch Generator
A Python script that uses [Stable Diffusion XL](https://stability.ai/stable-image) (SDXL) to generate batches of images based on your prompt.
I made this script around the time SDXL first came out and decided to upload it now.
- Supports SDXL base 1.0 and SDXL refiner 1.0 models
- Stable Diffusion Video (SVD) and SVD XT are implemented but not currently working
- Prompt weighting is supported with the syntax: "(token_string)weight_float"
- You can optionatly load LoRAs for use with the base model
- The [`diffusers`](https://github.com/huggingface/diffusers) library is set to offline mode in the script, so you have to manually download models
- Tested and working on Linux Mint 21.3 with a NVIDIA GTX 1070 (8GB VRAM)

**Copyright 2024, Benjamin Steenkamer - see [`LICENSE`](LICENSE) for details**

# Types of Batch Runs
Run different types of image generation batches by commenting/uncommenting calls in the `__main__` function.
All the main configuration variables (prompts, number of inference steps, number of images, etc.) are set in `__main__` and you can change them as needed.
This script supports:
- SDXL base model image generation, **with or without LoRAs**
- SDXL image in-painting: load an input and image mask, white areas of the mask will be transformed in the input image, black areas will be kept the same
- Image to Image conversion: load all the images in the output folder and transform into new images
- SDXL refiner model image generation to make refined images that are closer to the original prompt than with just the base model alone
- SDXL refiner model image generation to make images with more fine details

# VRAM Limitations
- This script is optimized to run on "low" VRAM GPUs like the NVIDIA GTX 1070 with *only* 8 GB of VRAM
- If you have more VRAM than this, you can enable an extra feature to improve processing speed: comment out the line `enable_model_cpu_offload()` and uncomment `to("cuda")`
- I do not know what will happen if you try running this script with less than 8 GB of VRAM

# File Locations
- `models/`: Place SDXL models here, each model must be in its own sub-folder
- `models/lora/`: Place LoRA safe tensor files here
- `outputs/`: Finished and latent (intermediate step) images will be saved here; output name is `seed-random_uuid.png`

# Model Links
- [SDXL Base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main): Download all folders and `model_index.json`; you don't need the 3 big safetensors
- [SDXL Refiner 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/tree/main): Download all folders and `model_index.json`; you don't need the 2 big safetensors

# Requirements
```
python==3.11
pillow
tqdm
torch
diffusers
compel
accelerate
omegaconf
xformers
peft
```
