# make sure you're logged in with `huggingface-cli login`
from torch import autocast
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)
device = 'cuda'
pipe = pipe.to(device)

prompt = "a photo of an astronaut riding a horse on mars"
with autocast(device):
    image = pipe(prompt).images[0]