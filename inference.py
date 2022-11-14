import onnxruntime as ort
import numpy as np
import os
from einops import rearrange
from PIL import Image
import math
from rich.progress import track
from transformers import CLIPTokenizer
import random

model_path = "models"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
unet = ort.InferenceSession(os.path.join(model_path, 'unet_fp16_opt.onnx'),  providers=providers)
decoder = ort.InferenceSession(os.path.join(model_path, 'decoder_fp16_opt.onnx'),  providers=['CPUExecutionProvider'])
post_quant_conv = ort.InferenceSession(os.path.join(model_path, 'post_quant_conv_fp16.onnx'),  providers=['CPUExecutionProvider'])
model_sigmas = np.load(os.path.join(model_path, 'sigmas.npy'))
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
transformer = ort.InferenceSession(os.path.join(model_path, 'transformer_fp16_opt.onnx'),  providers=['CPUExecutionProvider'])

height = 96
width = 64
seed = random.randint(1000000, 9999999)
print("seed:", seed)
steps = 28
cond_scale = 11

def get_sigmas(steps):
  t = np.linspace(len(model_sigmas) - 1, 0, steps)
  low_idx, high_idx, w = np.floor(t).astype(int), np.ceil(t).astype(int), np.modf(t)[0]
  x = (1 - w) * model_sigmas[low_idx] + w * model_sigmas[high_idx]
  return np.concatenate([x, np.zeros([1])]), t

def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = np.sqrt(np.square(sigma_to) * (np.square(sigma_from) - np.square(sigma_to)) / np.square(sigma_from))
    sigma_down = np.sqrt(np.square(sigma_to) - np.square(sigma_up))
    return sigma_down, sigma_up

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)

def get_scalings(sigma):
  c_out = -sigma
  c_in = 1 / (sigma ** 2 + 1.0 ** 2) ** 0.5
  return c_out, c_in

def sample(latent, sigma, timestep, condition, cond_scale):
  c_out, c_in = [append_dims(x, latent.ndim) for x in get_scalings(sigma)]
  latent_input = np.array(latent * c_in).astype(np.float16)
  latent_input_two = np.concatenate([latent_input] * 2)
  timestep_two = np.concatenate([timestep] * 2)
  uncond, cond = unet.run(["output"], {'input': latent_input_two, 'time_embedding': timestep_two, 'condition': condition})[0]
  x_0 = uncond + (cond - uncond) * cond_scale
  return latent + x_0 * c_out

def sample_euler_ancestral(image_latents, sigmas, timesteps, cond, cond_scale):
  s_in = np.array([1 for i in range(image_latents.shape[0])]).astype(np.float16)
  for i in track(range(len(sigmas) - 1), description="Sampling..."):
    denoised = sample(image_latents, sigmas[i], timesteps[i] * s_in, cond, cond_scale)
    sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
    d = to_d(image_latents, sigmas[i], denoised)
    dt = sigma_down - sigmas[i]
    image_latents = image_latents + d * dt
    image_latents = image_latents + np.random.normal(size=image_latents.shape).astype(np.float16) * sigma_up
  return image_latents

def getCond(text):
  tokens = tokenizer(text, truncation=True, max_length=77, return_length=True, return_overflowing_tokens=False, padding="max_length")["input_ids"]
  tokens = np.array([tokens]).astype(np.int64)
  z = transformer.run(["output"], {"input_ids": tokens})[0]
  return z

condition = getCond("1girl, white hair, green eye, kawaii, loli, short hair, bangs, no-highlights, white short shirt, pleated skirt, white skirt, black bow on chest, white socks, younger, young girl, best quality, masterpiece, standing")
uncondition = getCond("lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name")

cond_full = np.concatenate([uncondition, condition])
np.random.seed(seed)
sigmas, timesteps = get_sigmas(steps)
sigmas = sigmas.astype(np.float16)
image_latents = np.expand_dims(np.random.normal(size=[4, height, width]), 0).astype(np.float16)
image_latents *= sigmas[0]

image_latents = sample_euler_ancestral(image_latents, sigmas, timesteps, cond_full, cond_scale)

del sigmas
del condition

image_latents = 1. / 0.18215 * image_latents
image_latents = post_quant_conv.run(["output"], {"input": image_latents})[0]
results = decoder.run(["output"], {"input": image_latents})[0]
results = np.clip((results + 1.0) / 2.0, 0.0, 1.0)
for result in results:
    result = (255. * rearrange(result, 'c h w -> h w c')).astype(np.uint8)
    result = np.ascontiguousarray(result)
    img = Image.fromarray(result)
    img.show()