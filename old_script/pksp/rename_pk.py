import os
import json
from PIL import Image
import re
from PIL.PngImagePlugin import PngInfo
from fastprogress import progress_bar

from torchvision import transforms
import torch 
from transformers import AutoModelForImageSegmentation

import rembg 



def remove_background(img):
    # Load the image
    img = img.convert("RGBA")
    pixels = img.load()

    # Get image dimensions
    width, height = img.size

    # Collect boundary colors
    boundary_colors = []
    for x in range(width):
        boundary_colors.append(pixels[x, 0])
        boundary_colors.append(pixels[x, height - 1])
    for y in range(height):
        boundary_colors.append(pixels[0, y])
        boundary_colors.append(pixels[width - 1, y])

    # Find the most common boundary color
    background_color = max(set(boundary_colors), key=boundary_colors.count)

    # Remove background
    for x in range(width):
        for y in range(height):
            if pixels[x, y][:3] == background_color[:3]:
                pixels[x, y] = (0, 0, 0, 0)

    return img

def clean_pokemon_name(name):
    """Clean pokemon name to match dictionary keys"""
    # Replace hyphen with nothing (for cases like ho-oh -> hooh)
    #name = name.replace('-', '')
    return name

def get_next_available_filename(base_path):
    """Handle filename collisions by adding letters a, b, c, etc."""
    if not os.path.exists(base_path):
        return base_path
    
    name, ext = os.path.splitext(base_path)
    
    for suffix in ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'):
        new_path = f"{name}{suffix}{ext}"
        if not os.path.exists(new_path):
            return new_path
    
    raise Exception("Too many duplicate filenames")

def limit_colors(
        image,
        limit: int=16,
        color_palette=None,
        quantize: Image.Quantize=Image.Quantize.MEDIANCUT,
        dither: Image.Dither=Image.Dither.NONE,
        use_k_means: bool=False
    ):
    alpha = None
    if image.mode == "RGBA":
        alpha = image.getchannel("A")
        image = image.convert("RGB")

    if use_k_means:
        k_means_value = limit
    else:
        k_means_value = 0

    color_palette0 = image.quantize(colors=limit, kmeans=k_means_value, method=quantize, dither=Image.Dither.NONE)
    new_image = image.quantize(palette=color_palette0, dither=dither)

    if color_palette:
        #image = new_image.convert("RGB")
        new_image = image.quantize(palette=color_palette, dither=dither)

    if alpha:
        new_image = new_image.convert("RGBA")
        new_image.putalpha(alpha)

    return new_image

def create_grid(base_number, input_folder, output_folder):
    # Parameters
    image_size = 96
    grid_columns = 20
    white_image = Image.new("RGBA", (image_size, image_size), (255, 255, 255, 0))

    # Create big grid images
    #for base_number in range(0, 502):
    images = []
    for i in progress_bar(range(0, 502)):  # Collect images in sequence
        file_name = f"{base_number}.{i}.png"
        file_path = os.path.join(input_folder, file_name)
        if os.path.exists(file_path):
            images.append(Image.open(file_path))
        else:
            images.append(white_image.copy())

    #if not any(image != white_image for image in images):
    #    continue  # Skip grids with all white images

    rows = -(-len(images) // grid_columns)  # Calculate required rows
    grid_width = grid_columns * image_size
    grid_height = rows * image_size
    grid_image = Image.new("RGBA", (grid_width, grid_height), (255, 255, 255, 0))

    for idx, img in enumerate(images):
        x = (idx % grid_columns) * image_size
        y = (idx // grid_columns) * image_size
        #grid_image.paste(img, (x, y))
        grid_image.paste(img.resize((image_size, image_size), Image.Resampling.NEAREST), (x, y))

    grid_image.save(os.path.join(output_folder, f"{base_number}.png"))

def process_images(input_folder, output_folder, fusiondex_file, palette=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with open(fusiondex_file, 'r') as f:
        fusiondex = json.load(f)
    
    # Get list of PNG files first
    png_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]
    
    if 1:
        birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
        birefnet.to('cuda')
        birefnet.eval()
        birefnet.half()
        transform_image = transforms.Compose([
            transforms.Resize((896, 896)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        def remove_bg(img):
            img = img.convert("RGB")
            input_images = transform_image(img).unsqueeze(0).to('cuda').half()

            with torch.no_grad():
                pred = birefnet(input_images)[-1].sigmoid().cpu()[0].squeeze()

            pred_pil = transforms.ToPILImage()(pred)

            resized_img = img.resize((96, 96), Image.Resampling.NEAREST)
            pred_pil = pred_pil.resize((96, 96), Image.Resampling.LANCZOS)
            pred_binary = pred_pil.point(lambda p: 255 if p > 128 else 0, mode="1")

            combined_rgba = Image.composite(
                resized_img.convert("RGBA"),
                Image.new("RGBA", resized_img.size, (0, 0, 0, 0)),
                pred_binary
            )
            return combined_rgba
    else:
        def remove_bg(img):
            img = img.convert("RGB")
            output_img = rembg.remove(img)
            resized_img = output_img.resize((96, 96), Image.Resampling.NEAREST)

            return resized_img
    
    if palette is not None:
        palette = Image.open(palette)
        palette_image = palette
        ppalette = palette.getcolors()
        if ppalette:
            color_palette = palette.quantize(colors=len(list(set(ppalette))))
        else:
            colors = len(palette_image.getcolors()) if palette_image.getcolors() else 256
            color_palette = palette_image.quantize(colors, kmeans=colors)

    # Create progress bar
    pbar = progress_bar(png_files)
    
    # Store any errors to report at the end
    errors = []
    
    for filename in pbar:
        image_path = os.path.join(input_folder, filename)
        with Image.open(image_path) as img:
            png_info = ''
            if 'Parameters' in img.info:
                png_info = img.info['Parameters']
            elif 'Comment' in img.info:
                png_info = img.info['Comment']
            else:
                for k, v in img.info.items():
                    if isinstance(v, str) and 'fusion' in v.lower():
                        png_info = v
                        break
            
            if not png_info:
                errors.append(f"No fusion info found in {filename}")
                continue

            match = re.search(r'fusion (\S+)_body in (\S+)_style', png_info.lower())
            
            if match:
                body_pokemon = (match.group(1))
                style_pokemon = (match.group(2))
                
                body_id = fusiondex.get(body_pokemon)
                style_id = fusiondex.get(style_pokemon)
                
                if body_id is None or style_id is None:
                    base_output_path = os.path.join(output_folder, filename)
                else:
                    base_output_path = os.path.join(output_folder, f"{style_id}.{body_id}.png")
                output_path = get_next_available_filename(base_output_path)
                
                img = remove_bg(img)
                #if palette is not None:
                img = limit_colors(
                            img,
                            limit=16,
                            color_palette=color_palette,
                            #quantize=Image.Quantize.MEDIANCUT,
                            quantize=Image.Quantize.MAXCOVERAGE,
                            dither=Image.Dither.NONE,
                            use_k_means=True,
                        )

                fc = 1
                resized_img = img.resize((96*fc, 96*fc), Image.Resampling.NEAREST)
                resized_img.save(output_path, "PNG")
            else:
                errors.append(f"Fusion pattern not found in PNG info: {png_info}")
        
    
    # Print errors at the end
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"- {error}")

# Example usage
if __name__ == "__main__":
    fusiondex_file = r"old_script\pksp\swapped_fusiondex_links.json"

    base_number = 428
    base_dir = r"G:\Teng\generated"
    input_folder = os.path.join(base_dir,f"input{base_number}")
    output_folder = os.path.join(base_dir,f"{base_number}")
    palette = r"G:\Teng\pksp\smogon\img\1_gen5\glalie-gen5.png"
    
    process_images(input_folder, output_folder, fusiondex_file, palette)
    create_grid(base_number,output_folder,output_folder)