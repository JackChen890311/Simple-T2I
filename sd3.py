import os
import torch
from diffusers import StableDiffusion3Pipeline

# For downloading models from Hugging Face Hub
# from huggingface_hub import login
# with open('token.txt', 'r') as f:
#     token = f.read()
# login(token = token)

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe.to("cuda")

step = int(input('Stable Diffusion 3 Medium started! Enter "exit" to quit.\nPlease set how many step you want (type 28 for default): '))
guide = 7.0
prompt = input('Prompt: ')
os.makedirs('output', exist_ok=True)

while prompt != "exit":
    image = pipe(prompt=prompt, negative_prompt="", num_inference_steps=step, guidance_scale=guide).images[0]
    while os.path.exists(f'output/sd3_medium_{prompt}.png'):
        prompt += '_'
    image.save(f'output/sd3_medium_{prompt}.png')
    print(f'Image saved to output/sd3_medium_{prompt}.png')
    print('============================')
    prompt = input('Prompt: ')