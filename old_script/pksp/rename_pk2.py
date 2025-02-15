import os
import re
import json
from PIL import Image
import torch
from fastprogress import progress_bar

class ImageProcessingPipeline:
    def __init__(self, use_birefnet=True):
        self.use_birefnet = use_birefnet

        if use_birefnet:
            from torchvision import transforms
            from transformers import AutoModelForImageSegmentation
            self.birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
            self.birefnet.to('cuda').eval().half()
            self.transform_image = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            def remove_bg( img):
                img = img.convert("RGB")
                input_images = self.transform_image(img).unsqueeze(0).to('cuda').half()
                with torch.no_grad():
                    pred = self.birefnet(input_images)[-1].sigmoid().cpu()[0].squeeze()
                pred_pil = transforms.ToPILImage()(pred)
                pred_pil = pred_pil.resize((96, 96), Image.Resampling.LANCZOS)
                pred_binary = pred_pil.point(lambda p: 255 if p > 128 else 0, mode="1")
                combined_rgba = Image.composite(
                    img.resize((96, 96), Image.Resampling.NEAREST).convert("RGBA"),
                    Image.new("RGBA", (96, 96), (0, 0, 0, 0)),
                    pred_binary
                )
                return combined_rgba
            
        else:
            from rembg  import remove
            def remove_bg(img):
                img = img.convert("RGB")
                output_img = remove(img)
                combined_rgba = output_img.resize((96, 96), Image.Resampling.NEAREST)
                return combined_rgba
            
        self.remove_bg = remove_bg

    def get_next_available_filename(self, base_path):
        if not os.path.exists(base_path):
            return base_path

        name, ext = os.path.splitext(base_path)
        for suffix in ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'):
            new_path = f"{name}{suffix}{ext}"
            if not os.path.exists(new_path):
                return new_path

    def limit_colors(self, image, limit=16, color_palette=None, quantize=Image.Quantize.MAXCOVERAGE, dither=Image.Dither.NONE, use_k_means=True):
        alpha = None
        if image.mode == "RGBA":
            alpha = image.getchannel("A")
            image = image.convert("RGB")

        k_means_value = limit if use_k_means else 0
        color_palette0 = image.quantize(colors=limit, kmeans=k_means_value, method=quantize, dither=Image.Dither.NONE)
        new_image = image.quantize(palette=color_palette0, dither=dither)
        
        if color_palette:
            image = new_image.convert("RGB")
            new_image = image.quantize(palette=color_palette, dither=dither)


        if alpha:
            new_image = new_image.convert("RGBA")
            new_image.putalpha(alpha)

        return new_image



    def process_images(self, input_folder, output_folder, fusiondex_file, palette=None):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        with open(fusiondex_file, 'r') as f:
            fusiondex = json.load(f)

        png_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]

        color_palette = None
        if palette:
            palette_image = Image.open(palette)
            ppalette = palette_image.getcolors()
            colors = len(ppalette) if ppalette else 256
            color_palette = palette_image.quantize(colors=colors)

        for filename in progress_bar(png_files):
            image_path = os.path.join(input_folder, filename)
            with Image.open(image_path) as img:
                png_info = next((v for k, v in img.info.items() if isinstance(v, str) and 'fusion' in v.lower()), '')
                if not png_info:
                    continue

                match = re.search(r'fusion (\S+)_body in (\S+)_style', png_info.lower())
                if match:
                    body_id = fusiondex.get(match.group(1))
                    style_id = fusiondex.get(match.group(2))
                    base_output_path = os.path.join(output_folder, f"{style_id}.{body_id}.png") if body_id and style_id else os.path.join(output_folder, filename)
                else:
                    base_output_path = os.path.join(output_folder, filename)

                output_path = self.get_next_available_filename(base_output_path)
                img = self.remove_bg(img)
                img = self.limit_colors(img, limit=16, color_palette=color_palette, use_k_means=True)
                img.resize((96, 96), Image.Resampling.NEAREST).save(output_path, "PNG")

    def create_grid(self, base_number, input_folder, output_folder):
        image_size, grid_columns = 96, 20
        white_image = Image.new("RGBA", (image_size, image_size), (255, 255, 255, 0))

        images = [Image.open(os.path.join(input_folder, f"{base_number}.{i}.png")) if os.path.exists(os.path.join(input_folder, f"{base_number}.{i}.png")) else white_image.copy() for i in range(502)]
        rows = -(-len(images) // grid_columns)
        grid_image = Image.new("RGBA", (grid_columns * image_size, rows * image_size), (255, 255, 255, 0))

        for idx, img in enumerate(images):
            x, y = (idx % grid_columns) * image_size, (idx // grid_columns) * image_size
            grid_image.paste(img.resize((image_size, image_size), Image.Resampling.NEAREST), (x, y))

        grid_image.save(os.path.join(output_folder, f"{base_number}.png"))

if __name__ == "__main__":
    fusiondex_file = r"old_script\pksp\swapped_fusiondex_links.json"
    base_number = 482
    base_dir = r"G:\Teng\generated"
    input_folder = os.path.join(base_dir, f"input{base_number}")
    output_folder = os.path.join(base_dir, f"{base_number}")
    palette = r"G:\Teng\pksp\smogon\img\1_gen5\fennekin-gen5.png"

    
    #input_folder = r"G:\Teng\generated\bb"
    #output_folder = r"G:\Teng\generated\outputbb"
    #palette is None

    pipeline = ImageProcessingPipeline(use_birefnet=True)
    pipeline.process_images(input_folder, output_folder, fusiondex_file, palette)
    pipeline.create_grid(base_number, output_folder, output_folder)
