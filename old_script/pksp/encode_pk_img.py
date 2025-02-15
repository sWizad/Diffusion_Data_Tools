import os
import cv2
import torch
import numpy as np
import shutil
from fastprogress import progress_bar
from diffusers import AutoencoderKL
from typing import Tuple, Optional, Dict, List
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re


BUG = [10, 11, 12, 13, 14, 15, 46, 47, 48, 49, 123, 127, 165, 166, 167, 168, 193, 204, 205, 212, 213, 214, 270, 289, 303, 304, 348, 356, 363, 374, 382, 412, 419, 422, 423, 424, 474, 475]
DARK = [197, 198, 215, 228, 229, 248, 256, 262, 295, 310, 330, 331, 338, 347, 361, 375, 376, 377, 399, 409, 410, 421, 436, 437, 458, 487, 493, 494]
DRAGON = [147, 148, 149, 230, 297, 298, 299, 334, 336, 342, 343, 344, 345, 349, 350, 351, 368, 375, 376, 377, 378, 379, 393, 395, 396, 414, 415, 425, 426, 440, 441, 443, 444, 445, 446, 470, 471, 472, 473]
ICE = [87, 91, 124, 131, 144, 215, 220, 221, 225, 238, 262, 272, 274, 351, 427, 428, 429, 448, 461, 462]
GRASS = [1, 2, 3, 43, 44, 45, 46, 47, 69, 70, 71, 102, 103, 114, 152, 153, 154, 182, 187, 188, 189, 191, 192, 251, 266, 271, 276, 277, 278, 301, 302, 316, 317, 318, 352, 355, 360, 364, 400, 401, 404, 408, 413, 438, 439, 453, 457, 458, 476, 477, 479, 480, 481, 489, 490, 495, 496, 497]
FIRE = [4, 5, 6, 37, 38, 58, 59, 77, 78, 126, 136, 146, 155, 156, 157, 218, 219, 228, 229, 240, 244, 250, 268, 279, 280, 281, 319, 320, 321, 349, 365, 366, 367, 372, 374, 418, 419, 430, 482, 483, 484, 488]
WATER = [7, 8, 9, 54, 55, 60, 61, 62, 72, 73, 79, 80, 86, 87, 90, 91, 98, 99, 116, 117, 118, 119, 120, 121, 129, 130, 131, 134, 138, 139, 140, 141, 158, 159, 160, 170, 171, 183, 184, 186, 194, 195, 199, 211, 222, 223, 224, 226, 230, 245, 261, 282, 283, 284, 314, 322, 323, 324, 335, 340, 344, 370, 383, 387, 394, 436, 437, 454, 455, 469, 474, 475, 485, 486, 487, 495, 496, 497, 501]
GROUND = [27, 28, 31, 34, 50, 51, 74, 75, 76, 95, 104, 105, 111, 112, 194, 195, 207, 208, 220, 221, 231, 232, 246, 247, 265, 273, 274, 283, 284, 297, 298, 299, 318, 334, 341, 361, 369, 382, 392, 393, 409, 410, 416, 420, 459, 460]
GHOST = [92, 93, 94, 200, 255, 289, 295, 311, 312, 313, 327, 328, 329, 345, 353, 357, 358, 362, 365, 366, 367, 369, 373, 402, 405, 411, 416, 421, 429, 433, 438, 439, 453, 459, 460, 489, 490]
FLYING = [6, 12, 16, 17, 18, 21, 22, 41, 42, 83, 84, 85, 123, 130, 142, 144, 145, 146, 149, 163, 164, 165, 166, 169, 176, 177, 178, 187, 188, 189, 193, 198, 207, 225, 226, 227, 249, 250, 256, 261, 269, 270, 273, 336, 342, 353, 356, 372, 402, 417, 418, 430, 431, 432, 433, 440, 441, 442, 443, 456, 498, 499]
FAIRY = [35, 36, 39, 40, 122, 173, 174, 175, 176, 183, 184, 209, 210, 252, 258, 269, 285, 286, 287, 300, 339, 360, 371, 373, 408, 478, 491, 492, 500]
FIGHTING = [56, 57, 62, 66, 67, 68, 106, 107, 214, 236, 237, 280, 281, 288, 296, 320, 321, 355, 384, 451, 452, 456, 467, 472, 473, 481, 493, 494]
ELECTRIC = [25, 26, 81, 82, 100, 101, 125, 135, 145, 170, 171, 172, 179, 180, 181, 239, 243, 263, 267, 332, 350, 358, 363, 388, 389, 412, 420, 431]
STEEL = [81, 82, 205, 208, 212, 227, 263, 291, 292, 293, 296, 300, 307, 308, 324, 326, 327, 328, 329, 330, 331, 333, 337, 343, 348, 364, 371, 381, 390, 391, 397, 398, 413, 449]
ROCK = [74, 75, 76, 95, 111, 112, 138, 139, 140, 141, 142, 185, 213, 219, 222, 246, 247, 248, 257, 265, 301, 302, 303, 304, 305, 306, 307, 308, 325, 326, 333, 390, 391, 425, 426, 447, 461, 462, 463, 464, 465, 478, 498, 499, 500]
PSYCHIC = [63, 64, 65, 79, 80, 96, 97, 102, 103, 121, 122, 124, 150, 151, 177, 178, 196, 199, 201, 202, 203, 238, 249, 251, 253, 258, 285, 286, 287, 288, 291, 292, 293, 359, 378, 379, 380, 381, 406, 407, 432, 450, 466, 468, 469, 470, 484]
POISON = [1, 2, 3, 13, 14, 15, 23, 24, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 48, 49, 69, 70, 71, 72, 73, 88, 89, 92, 93, 94, 109, 110, 167, 168, 169, 211, 352, 400, 401, 422, 423, 424, 434, 435, 454, 455]
NORMAL = [16, 17, 18, 19, 20, 21, 22, 39, 40, 52, 53, 83, 84, 85, 108, 113, 115, 128, 132, 133, 137, 143, 161, 162, 163, 164, 174, 190, 203, 206, 216, 217, 233, 234, 235, 241, 242, 252, 254, 259, 260, 264, 275, 290, 294, 309, 315, 346, 354, 383, 385, 386, 403, 417, 442, 451, 452, 466, 467]
ALL = [i for i in range(1,505)]
LEAS = [486, 493, 480, 496, ]


class ImageEncoderPipeline:
    TARGET_SIZES = {
        '1024': [(768, 768),]
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
            
    
    def get_image_files(self, folder: str) -> List[str]:
        image_files = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    image_files.append(os.path.join(root, file))
        return image_files
    
    def process_image(self, img: np.ndarray, speed_scale: float = 1, visibility: float = 0.75) -> np.ndarray:
        white_bg = np.ones_like(img[:, :, :3]) * 255
        if img.shape[-1] == 4:
            alpha = img[:, :, 3:4] / 255.0
            rgb = img[:, :, :3]
            
            blended = rgb * alpha + white_bg * (1 - alpha)
            img = blended.astype(np.uint8)
        else:
            img = img[:, :, :3].astype(np.uint8)

        if self.upsampler is None:
            small = cv2.resize(img, (96, 96), interpolation=cv2.INTER_NEAREST)
            final = cv2.resize(small, (768, 768), interpolation=cv2.INTER_NEAREST)
            return final
            

    
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
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            
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

class ImageBatchDataset(Dataset):
    def __init__(self, image_files: List[str], target_sizes: Dict, speed_scale: float, visibility: float):
        self.image_files = image_files
        self.target_sizes = target_sizes
        self.speed_scale = speed_scale
        self.visibility = visibility
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  
        
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8) 

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

        aspect_ratio = img.shape[1] / img.shape[0]
        
        return {
            'image': img_rgb,
            'path': image_path,
            'aspect_ratio': aspect_ratio
        }

class BatchImageEncoderPipeline(ImageEncoderPipeline):
    TARGET_SIZES = {
        '1024': [(768, 768),]
    }
    
    def __init__(self, device: str = 'cuda', batch_size: int = 1):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.upsampler = None
        self.vae = None
    
    def get_image_files_old(self, folder: str,) -> List[str]:
        image_files = []
        pattern = re.compile(r'^(\d+)\.(\d+)[a-zA-Z]?\.png$')
        
        for root, _, files in os.walk(folder):
            for file in files:
                match = pattern.match(file)
                if match and int(match.group(1)) in self.valid_numbers:
                    image_files.append(os.path.join(root, file))
        
        return image_files
    
    def get_image_files(self, folder: str, output_folder: str) -> List[str]:
        image_files = []
        pattern = re.compile(r'^(\d+)\.(\d+)[a-zA-Z]?\.png$')
        
        for root, _, files in os.walk(folder):
            for file in files:
                match = pattern.match(file)
                if match and int(match.group(1)) in self.valid_numbers:
                    # Generate the corresponding output path
                    rel_path = os.path.relpath(os.path.join(root, file), folder)
                    output_path = os.path.join(output_folder, rel_path)
                    
                    # Only add the file if it doesn't exist in the output folder
                    if not os.path.exists(output_path):
                        image_files.append(os.path.join(root, file))
        
        return image_files

    def generate_latents_batch_old(self, images: List[np.ndarray]) -> torch.Tensor:
        # Normalize and convert batch of images to tensor
        images_norm = [(img / 255.0) * 2 - 1 for img in images]
        image_tensors = torch.stack([
            torch.from_numpy(np.transpose(img, (2, 0, 1))).float() 
            for img in images_norm
        ]).to(self.device)
        
        with torch.no_grad():
            latents = self.vae.encode(image_tensors).latent_dist.sample()
            return latents
        
    def generate_latents_batch(self, images: List[np.ndarray]):
        return [None] * len(images)

    def save_latents(self, path: str, latents: torch.Tensor, original_size: Tuple[int, int], 
                    crop_ltrb: Tuple[int, int, int, int]) -> None:
        np.savez(path, latents=latents.float().cpu().numpy(),
                original_size=np.array(original_size),
                crop_ltrb=np.array(crop_ltrb))
        
    def save_black_image(self, height, width, output_path):
        black_image = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.imwrite(output_path, black_image)

    def process_folder(self, input_folder: str, output_folder: str, speed_scale: float = 0.8, 
                      visibility: float = 0.75, is_save_black = False) -> None:
        os.makedirs(output_folder, exist_ok=True)
        image_files = self.get_image_files(input_folder,output_folder)
        
        # Create dataset and dataloader for batch processing
        dataset = ImageBatchDataset(image_files, self.TARGET_SIZES, speed_scale, visibility)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4, 
                              shuffle=False, pin_memory=True)
        
        for batch in progress_bar(dataloader):
            batch_images = batch['image']
            batch_paths = batch['path']
            batch_aspect_ratios = batch['aspect_ratio']
            
            # Process batch of images
            processed_images = []
            original_sizes = []
            crop_ltrbs = []
            
            for i in range(len(batch_images)):
                target_size = self.get_target_size(batch_aspect_ratios[i].item())
                processed = self.process_image(batch_images[i].numpy(), speed_scale, visibility)
                image, original_size, crop_ltrb = self.trim_and_resize(processed, target_size)
                processed_images.append(image)
                original_sizes.append(original_size)
                crop_ltrbs.append(crop_ltrb)
            
            # Generate latents for batch
            batch_latents = self.generate_latents_batch(processed_images)
            
            # Save results for each image in batch
            for i, (image_path, processed_image, latent) in enumerate(zip(batch_paths, processed_images, batch_latents)):
                relative_path = os.path.relpath(image_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)
                latent_path = os.path.join(output_folder, f'{os.path.splitext(relative_path)[0]}.npz')
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                if is_save_black:
                    self.save_black_image(processed_image.shape[0], processed_image.shape[1], output_path)
                else:
                    output = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_path, output)
                
                if latent is not None:
                    self.save_latents(latent_path, latent, original_sizes[i], crop_ltrbs[i])
                
                # Copy associated text file if it exists
                txt_path = os.path.splitext(image_path)[0] + '.txt'
                if os.path.exists(txt_path):
                    txt_output_path = os.path.join(output_folder, 
                                                 os.path.relpath(txt_path, input_folder))
                    os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)
                    shutil.copy2(txt_path, txt_output_path)

if __name__ == "__main__":
    pipeline = BatchImageEncoderPipeline()
    pipeline.load_models(
        model_mode=None,
    )
    pipeline.valid_numbers = ALL
    pipeline.process_folder(
        r'G:\Teng\Full Sprite pack 1-108 (November 2024)\CustomBattlers', 
        r'G:\Teng\Full Sprite pack 1-108 (November 2024)\OutForAll',
        is_save_black = False,
        speed_scale = 0.8,
        )