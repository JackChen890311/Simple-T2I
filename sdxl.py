import os
import torch
from diffusers import DiffusionPipeline

# Load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    use_safetensors=True,
    variant="fp16", 
)
base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Speed up
base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

print('Stable Diffusion XL started! Enter "exit" to quit.')
prompt = input('Prompt: ')
os.makedirs('output', exist_ok=True)

while prompt != "exit":
    # Run both experts
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]

    # Save image
    while os.path.exists(f'output/{prompt}.png'):
        prompt += '_'
    image.save(f'output/{prompt}.png')
    print(f'Image saved to output/{prompt}.png')
    print('============================')
    prompt = input('Prompt: ')