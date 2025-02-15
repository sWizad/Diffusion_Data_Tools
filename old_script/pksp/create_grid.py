import os
from PIL import Image, ImageDraw
from fastprogress import progress_bar

from PIL import Image



if __name__ == "__main__":
    base_number = 428
    # Input folder and output folder
    input_folder = r"G:\Teng\Test"
    input_folder = os.path.join(input_folder,str(base_number))
    output_folder = r"G:\Teng\Test"

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Parameters
    image_size = 96
    grid_columns = 20
    white_image = Image.new("RGBA", (image_size, image_size), (255, 255, 255, 0))

    # Create big grid images
    #for base_number in range(0, 502):
    if 1:
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
            grid_image.paste(img.resize((96, 96), Image.Resampling.NEAREST), (x, y))

        grid_image.save(os.path.join(output_folder, f"{base_number}.png"))
