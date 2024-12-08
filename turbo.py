import os
import torch
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

step = int(input('Stable Diffusion XL Turbo started! Enter "exit" to quit.\nPlease set how many step you want: '))
guide = 0.0
prompt = input('Prompt: ')
os.makedirs('output', exist_ok=True)

while prompt != "exit":
    image = pipe(prompt=prompt, num_inference_steps=step, guidance_scale=guide).images[0]
    while os.path.exists(f'output/sdxl_turbo_{prompt}.png'):
        prompt += '_'
    image.save(f'output/sdxl_turbo_{prompt}.png')
    print(f'Image saved to output/sdxl_turbo_{prompt}.png')
    print('============================')
    prompt = input('Prompt: ')

