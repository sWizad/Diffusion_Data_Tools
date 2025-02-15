import os
import cv2
import torch
import numpy as np
import shutil
from fastprogress import progress_bar
from lib.RRDBNet_arch import RRDBNet, RealESRGANer
from diffusers import AutoencoderKL
from typing import Tuple, Optional, Dict, List

class ImageEncoderPipeline:
    TARGET_SIZES = {
        '512': [(512, 512), (448, 576), (576, 448)],
        '1024': [(1024, 1024), (896, 1152), (1152, 896)]
    }
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device)
        self.upsampler = None
        self.vae = None
        
    def load_models(self, 
                   vae_path: str = r"E:\Research\symlink\model\PonyV6XL_4base.safetensors",
                   model_mode: Optional[str] = None, 
                   upscaler_base_path: str =  r"E:\Research\stable-diffusion-webui-reForge\models\RealESRGAN", ) -> None:
        self.vae = AutoencoderKL.from_single_file(vae_path)
        self.vae.eval()
        self.vae.to(self.device)
        
        if model_mode is None:
            return
            
        if model_mode.lower() == 'real':
            model_path = os.path.join(upscaler_base_path, 'RealESRGAN_x4plus.pth')
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        elif model_mode.lower() == 'anime':
            model_path = os.path.join(upscaler_base_path, 'RealESRGAN_x4plus_anime_6B.pth')
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        
        model.to(self.device)
        self.upsampler = RealESRGANer(
            scale=4, model_path=model_path, model=model,
            tile=0, tile_pad=10, pre_pad=0, half=False, gpu_id=0
        )
    
    def get_image_files(self, folder: str) -> List[str]:
        image_files = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    image_files.append(os.path.join(root, file))
        return image_files
    
    def process_image(self, img: np.ndarray, speed_scale: float = 1, visibility: float = 0.75) -> np.ndarray:
        if self.upsampler is None:
            return img
            
        h_input, w_input = img.shape[0:2]
        img_input = cv2.resize(img, 
                             (int(w_input * speed_scale), int(h_input * speed_scale)),
                             interpolation=cv2.INTER_LANCZOS4)
        
        output, _ = self.upsampler.enhance(img_input)
        
        outscale = 1/speed_scale
        output = cv2.resize(output, (int(w_input * outscale), int(h_input * outscale)),
                          interpolation=cv2.INTER_LANCZOS4)
        
        if visibility == 1: return output
        img = cv2.resize(img, (int(w_input * outscale), int(h_input * outscale)),
                        interpolation=cv2.INTER_LANCZOS4)
                        
        return (output * visibility + img * (1-visibility)).round().astype(np.uint8)
    
    def get_target_size(self, aspect_ratio: float, max_size: int = 1024) -> Tuple[int, int]:
        def size_difference(size):
            target_aspect_ratio = size[0] / size[1]
            return abs(target_aspect_ratio - aspect_ratio)
            
        target_size = min(self.TARGET_SIZES[str(max_size)], key=size_difference)
        return target_size
    
    def trim_and_resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]:
        original_size = (image.shape[1], image.shape[0])
        
        if image.shape[1] != target_size[0] or image.shape[0] != target_size[1]:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        crop_left = (target_size[0] - image.shape[1]) // 2
        crop_top = (target_size[1] - image.shape[0]) // 2
        crop_right = crop_left + image.shape[1]
        crop_bottom = crop_top + image.shape[0]
        
        return image, original_size, (crop_left, crop_top, crop_right, crop_bottom)
    
    def generate_latent(self, image: np.ndarray) -> torch.Tensor:
        image_norm = (image / 255.0) * 2 - 1
        image_tensor = torch.from_numpy(np.transpose(image_norm, (2, 0, 1))[None, ...]).float()
        
        with torch.no_grad():
            latent = self.vae.encode(image_tensor.to(self.device)).latent_dist.sample()
            return latent[0]
    
    def save_latents(self, path: str, latents: torch.Tensor, original_size: Tuple[int, int], 
                    crop_ltrb: Tuple[int, int, int, int]) -> None:
        np.savez(path, latents=latents.float().cpu().numpy(),
                original_size=np.array(original_size),
                crop_ltrb=np.array(crop_ltrb))
        
    def save_black_image(self, height, width, output_path):
        black_image = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.imwrite(output_path, black_image)

    def process_folder(self, input_folder: str, output_folder: str, speed_scale: float = 0.8, visibility: float = 0.75, is_save_black = False) -> None:
        os.makedirs(output_folder, exist_ok=True)
        image_files = self.get_image_files(input_folder)
        
        for image_path in progress_bar(image_files):
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            aspect_ratio = img.shape[1] / img.shape[0]
            target_size = self.get_target_size(aspect_ratio)
            
            processed = self.process_image(img_rgb, speed_scale, visibility=visibility)
            image, original_size, crop_ltrb = self.trim_and_resize(processed, target_size)
            
            relative_path = os.path.relpath(image_path, input_folder)
            output_path = os.path.join(output_folder, relative_path)
            latent_path = os.path.join(output_folder, f'{os.path.splitext(relative_path)[0]}.npz')
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if is_save_black :
                self.save_black_image(image.shape[0], image.shape[1], output_path)
            else:
                output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, output)
            
            latent = self.generate_latent(image)
            self.save_latents(latent_path, latent, original_size, crop_ltrb)

            txt_path = os.path.splitext(image_path)[0] + '.txt'
            if os.path.exists(txt_path):
                txt_output_path = os.path.join(output_folder, 
                                             os.path.relpath(txt_path, input_folder))
                os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)
                shutil.copy2(txt_path, txt_output_path)

if __name__ == "__main__":
    pipeline = ImageEncoderPipeline()
    pipeline.load_models(
        #model_mode="real",
        model_mode="anime",
        #model_mode=None,
    )
    pipeline.process_folder(
        r'E:\Research\symlink\CivitAI\riot\arcane\img', 
        r'E:\Research\symlink\CivitAI\riot\arcane\img_up2',
        is_save_black = True,
        speed_scale = 0.8,
        )