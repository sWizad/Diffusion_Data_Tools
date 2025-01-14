import os
from PIL import Image
import glob
from concurrent.futures import ThreadPoolExecutor
from fastprogress import progress_bar

def process_single_image(image_path):
    try:
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            ratio = min(1024/img.size[0], 1024/img.size[1])
            new_size = (int(img.size[0]*ratio), int(img.size[1]*ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            background = Image.new('RGB', (1024, 1024), 'black')
            offset = ((1024 - new_size[0]) // 2, (1024 - new_size[1]) // 2)
            background.paste(img, offset)
            
            base_path = os.path.splitext(image_path)[0]
            output_path = f"{base_path}.png"
            
            # Save with maximum PNG compression
            #background.save(output_path, 'PNG', optimize=True, compress_level=9)
            background.save(output_path, 'PNG',)
            
            if not image_path.lower().endswith('.png'):
                os.remove(image_path)
            
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

def process_images(directory_path):
    supported_formats = ['.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
    image_paths = []
    
    for root, _, files in os.walk(directory_path):
        for format in supported_formats + ['.png']:
            pattern = os.path.join(root, f'*{format}')
            image_paths.extend(glob.glob(pattern))
    
    with ThreadPoolExecutor() as executor:
        list(progress_bar(executor.map(process_single_image, image_paths), total=len(image_paths)))


if __name__ == "__main__":
    # Get directory path from user
    directory = r"E:\Research\symlink\CivitAI\disney\realcartoon\draft-3"
    
    # Verify directory exists
    if os.path.isdir(directory):
        process_images(directory)
        print("Processing complete!")
    else:
        print("Invalid directory path!")