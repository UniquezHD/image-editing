import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16,
)

pipe.enable_model_cpu_offload()

prompt = "make her smile"

input_image = load_image("input.png")

image = pipe(
    image=input_image,
    prompt=prompt,
).images[0]

image.save("output.png")