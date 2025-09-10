import torch
from diffusers import FluxPipeline
from diffusers import FluxTransformer2DModel

# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, cache_dir="/data2/onkar/llava")
# pipe.enable_model_cpu_offload()

transformer = FluxTransformer2DModel.from_pretrained(pretrained_model_name_or_path="black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, cache_dir="/data2/onkar/llava", subfolder="transformer", use_2nd_guider=True)
# pipe.to("cuda")

# prompt = "white dog jumping on the ground"
# image = pipe(
#     prompt,
#     guidance_scale=12.0,
#     num_inference_steps=4,
#     max_sequence_length=32,
#     height=1024,
#     width=1024,
#     generator=torch.Generator("cuda")
# ).images[0]

# image.save("flux-schnell.png")
