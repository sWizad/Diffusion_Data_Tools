import cv2
import torch
from lib.RRDBNet_arch import RRDBNet, RealESRGANer
from fastprogress import progress_bar
import time, os
from diffusers import AutoencoderKL
import numpy as np


model_path = r'E:\Research\stable-diffusion-webui-reForge\models\RealESRGAN\RealESRGAN_x4plus.pth' 
device = torch.device('cuda') 

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model.to(device)
upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    dni_weight=None,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
    gpu_id=0)

ckpt = r"E:\Research\symlink\model\PonyV6XL_4base.safetensors"
vae = AutoencoderKL.from_single_file(ckpt)
vae.eval()
vae.to(device)

# Define input and output folders
input_folder = 'data/test_in'
output_folder = 'data/test_out'
latent_folder = output_folder
speed_scale = 0.8
os.makedirs(output_folder, exist_ok=True)

# Get all image files from the folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

# Start timer
start_time = time.time()

# Process each image
for image_file in progress_bar(image_files):
    input_path = os.path.join(input_folder, image_file)
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    h_input, w_input = img.shape[0:2]
    img = cv2.resize( img, (
                    int(w_input * speed_scale),
                    int(h_input * speed_scale),
                ), interpolation=cv2.INTER_LANCZOS4)
    output, _ = upsampler.enhance(img, outscale=1/speed_scale)
    
    output_path = os.path.join(output_folder, f'{image_file}')
    cv2.imwrite(output_path, output)

    output = output / 255.0  # Normalize image to [0, 1]
    output_tensor = np.transpose(output, (2, 0, 1))[None, ...]  # HWC -> NCHW
    output_tensor = torch.from_numpy(output_tensor).float()  # Convert to tensor
    
    with torch.no_grad():
        latent = vae.encode(output_tensor.to("cuda")).latent_dist.sample()
        latent_np = latent.cpu().numpy()  # Convert latent tensor to numpy
    
    # Save latent to .npz file
    latent_path = os.path.join(latent_folder, f'{os.path.splitext(image_file)[0]}.npz')
    np.savez(latent_path, latent=latent_np)

# Calculate and report total time
total_time = time.time() - start_time
print(f"Total running time: {total_time:.2f} seconds")
