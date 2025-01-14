from pathlib import Path
import numpy as np
from PIL import Image
from fastprogress import progress_bar
import os

def process_image_folder(input_folder, output_folder):
    """
    Process all images in the given folder:
    1. Pad images smaller than 96x96 to be centered in a white 96x96 image
    2. Resize all images from 96x96 to 768x768 using nearest neighbor interpolation
    3. Save with format: {original_name}-{folder_name}.png
    """
    print(input_folder)
    
    # Create input folder path object
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output folder if it doesn't exist
    #output_path = input_path / 'processed'
    output_path.mkdir(exist_ok=True)
    
    # Get list of image files
    image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.jpeg')) + list(input_path.glob('*.png'))
    
    # Get folder name for output filename
    folder_name = input_path.name
    
    # Process each image with progress bar
    for img_path in progress_bar(image_files):
        try:
            # Open image
            img = Image.open(img_path)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Check if padding is needed
            if img.size[0] < 96 or img.size[1] < 96:
                # Create new white image
                padded_img = Image.new('RGB', (96, 96), 'white')
                
                # Calculate padding
                left = (96 - img.size[0]) // 2
                top = (96 - img.size[1]) // 2
                
                # Paste original image in center
                padded_img.paste(img, (left, top))
                img = padded_img
            else:
                # Resize to 96x96 if larger
                img = img.resize((96, 96), Image.NEAREST)
            
            # Resize to final size
            img = img.resize((768, 768), Image.NEAREST)
            
            # Create output filename
            output_filename = f"{img_path.stem}-{folder_name}.png"
            output_file = output_path / output_filename
            
            # Save processed image
            img.save(output_file, 'PNG')
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue

if __name__ == "__main__":
    # Example usage
    name_list = [
        'gen1',
        'gen2',
        'gen3',  'gen3-frlg',   'gen3-rs',
        'gen4',  'gen4-dp',   'gen4-dp2',
        'gen5', 'trainers'
    ]
    for name in name_list:
        folder_path = Path(r"E:\Research\Diffusion_Data_Tools\data\pokemon_sprites") / name
        output_folder = Path(r"E:\Research\symlink\CivitAI\pksp\smogon\draft") / name
        process_image_folder(folder_path,output_folder)