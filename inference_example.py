import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from PIL import Image
checkpoint = 'path_to_model_file'
sketch_path = 'path_to_sketch_file'
controlnet = ControlNetModel.from_pretrained(
          checkpoint, subfolder='controlnet', torch_dtype=torch.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)
pipe.safety_checker = None
pipe.requires_safety_checker = False
pipe.to('cuda')

sketch_image = Image.open(sketch_path)
width, height = sketch_image.size

if width < height:
    new_width = 512
    new_height = int(height * (512 / width))
else:
  new_height = 512
  new_width = int(width * (512 / height))

sketch_image = sketch_image.resize((new_width, new_height))

condition_image = sketch_image

prompt = 'Traditional Thai Line Art'
  
image = pipe(
    prompt,
    condition_image,
    num_inference_steps=50,
).images[0]

image.save('output.png')
