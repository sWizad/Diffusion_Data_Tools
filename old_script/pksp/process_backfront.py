import os
from PIL import Image
import numpy as np
from pathlib import Path
from fastprogress import progress_bar

def get_common_files(folder1, folder2):
    """Find PNG files that exist in both folders."""
    files1 = set(f.lower() for f in os.listdir(folder1) if f.endswith('.png'))
    files2 = set(f.lower() for f in os.listdir(folder2) if f.endswith('.png'))
    return list(files1.intersection(files2))

def remove_transparency(img):
    # Create a white background
    bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
    # Paste the image on the background using alpha channel as mask
    return Image.alpha_composite(bg, img).convert('RGB')


def center_image_on_white(img, target_size=(96, 96)):
    """Center an image on a white background of target size without resizing."""
    # Convert to RGBA if not already
    img = img.convert('RGBA')
    
    # Create new white image
    white_bg = Image.new('RGBA', target_size, (255, 255, 255, 255))
    
    # Calculate position to paste the image centered
    paste_x = max(0, (target_size[0] - img.width) // 2)
    paste_y = max(0, (target_size[1] - img.height) // 2)
    
    # If image is larger than target size, we need to crop it
    if img.width > target_size[0] or img.height > target_size[1]:
        # Calculate crop coordinates
        crop_left = max(0, (img.width - target_size[0]) // 2)
        crop_top = max(0, (img.height - target_size[1]) // 2)
        crop_right = min(img.width, crop_left + target_size[0])
        crop_bottom = min(img.height, crop_top + target_size[1])
        
        # Crop the image
        img = img.crop((crop_left, crop_top, crop_right, crop_bottom))
        paste_x = max(0, (target_size[0] - img.width) // 2)
        paste_y = max(0, (target_size[1] - img.height) // 2)
    
    # Paste the image on the white background using alpha channel as mask
    white_bg.paste(img, (paste_x, paste_y), img)
    
    return white_bg.convert('RGB')

def process_image_pair(file_name, folder1, folder2, output_folder, suffix):
    """Process a pair of images according to specifications."""
    # Open images
    img1_path = os.path.join(folder1, file_name)
    img2_path = os.path.join(folder2, file_name)
    
    img1 = Image.open(img1_path).convert('RGBA')
    img2 = Image.open(img2_path).convert('RGBA')
    
    img1 = center_image_on_white(img1)
    img2 = center_image_on_white(img2)
    
    # Create new image 96x192
    combined = Image.new('RGB', (192, 96))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (96, 0))
    
    # Upscale 3x with nearest neighbor
    final_img = combined.resize((1536, 768), Image.Resampling.NEAREST)
    
    # Create output path
    base_name = os.path.splitext(file_name)[0]
    output_path = os.path.join(output_folder, f"{base_name}-{suffix}.png")
    
    # Save the final image
    final_img.save(output_path)
    return output_path

def main(folder1, folder2, output_folder, suffix):
    """Main function to process all common PNG files."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get common files
    common_files = get_common_files(folder1, folder2)
    
    if not common_files:
        print("No common PNG files found in both folders.")
        return
    
    # Process each pair of images
    for file_name in progress_bar(common_files):
        try:
            output_path = process_image_pair(file_name, folder1, folder2, output_folder, suffix)
            #print(f"Processed: {file_name} -> {output_path}")
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

if __name__ == "__main__":
    # Example usage
    folder1 = r"E:\Research\Diffusion_Data_Tools\data\pokemon_sprites\gen5"
    folder2 = r"E:\Research\Diffusion_Data_Tools\data\pokemon_sprites\gen5-back"
    output_folder = r"E:\Research\symlink\CivitAI\pksp\smogon\img\1_gen5-back"
    suffix = "gen5" 
    
    main(folder1, folder2, output_folder, suffix)