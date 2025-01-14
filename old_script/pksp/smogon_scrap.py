import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image
import io
import multiprocessing as mp
from functools import partial
import numpy as np
from fastprogress import progress_bar

def get_sprite_urls(base_url):
    """Scrape Pokemon sprite URLs from the given webpage"""
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        sprite_urls = [link.get('href') for link in soup.find_all('a') 
                      if link.get('href', '').endswith('.png')]
        print(f"Found {len(sprite_urls)} sprite URLs")
        return sprite_urls
    except requests.exceptions.RequestException as e:
        print(f"Error accessing {base_url}: {e}")
        return []

def process_image(img_data, bg_color=(255, 255, 255)):
    """Process image to replace transparent background with solid color"""
    # Open image from binary data
    img = Image.open(io.BytesIO(img_data))
    
    # Convert to RGBA if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Convert to numpy array for faster processing
    rgba = np.array(img)
    
    # Create new RGB array with background color
    rgb = np.zeros((rgba.shape[0], rgba.shape[1], 3), dtype=np.uint8)
    rgb[:] = bg_color
    
    # Calculate alpha factor for blending
    alpha = rgba[:, :, 3:4] / 255.0
    
    # Blend original RGB with background using alpha
    rgb = rgba[:, :, :3] * alpha + (1 - alpha) * rgb
    
    # Convert back to PIL Image
    return Image.fromarray(rgb.astype(np.uint8), 'RGB')

def download_and_process_sprite(args):
    """Download and process a single sprite (for multiprocessing)"""
    sprite_url, base_url, output_folder = args
    full_url = urljoin(base_url, sprite_url)
    filename = os.path.join(output_folder, sprite_url.split('/')[-1])
    
    try:
        # Download the image
        response = requests.get(full_url)
        response.raise_for_status()
        
        # Process the image
        processed_img = process_image(response.content)
        
        # Save the processed image
        processed_img.save(filename, 'PNG')
        
        return f"Successfully processed: {sprite_url}"
    except Exception as e:
        return f"Error processing {sprite_url}: {e}"

def download_pokemon_sprites(base_url, sprite_urls, output_folder, num_processes=None):
    """Download and process Pokemon sprites using multiprocessing with progress bar"""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # If num_processes is None, use CPU count
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Prepare arguments for multiprocessing
    args = [(url, base_url, output_folder) for url in sprite_urls]
    
    # Create pool and process images with progress bar
    print(f"Starting download and processing with {num_processes} processes...")
    with mp.Pool(num_processes) as pool:
        results = list(progress_bar(pool.imap(download_and_process_sprite, args), total=len(args),))
    
    # Print results
    successful = sum(1 for r in results if r.startswith("Successfully"))
    print(f"\nProcessed {successful} out of {len(sprite_urls)} sprites")
    
    # Print errors if any
    errors = [r for r in results if r.startswith("Error")]
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(error)

if __name__ == "__main__":
    # Define folder structure
    FOLDER = 'gen5-back'
    folder_list = [
        'gen5-shiny', 'gen5-back-shiny', 
        'gen4', 'gen4-back',
        'gen4-shiny', 'gen4-back-shiny',
        'gen4dp-2', 'gen4dp-2-shiny',
        'gen4dp', 'gen4dp-shiny',
    ]
    folder_list = [
        'gen3', 'gen3-back',
        'gen3-shiny', 'gen3-shiny-back',
        'gen3rs', 'gen3rs-shiny',
        'gen3frlg',
    ]
    folder_list = [
        'trainers',
        'gen1', 'gen1-back',
        'gen2', 'gen2-back',
        'gen2-shiny', 'gen2-back-shiny',
        'gen2-g', 'gen2-s'
    ]
    for FOLDER in folder_list:
        BASE_URL = f"https://play.pokemonshowdown.com/sprites/{FOLDER}/"
        OUTPUT_FOLDER = f"data/pokemon_sprites/{FOLDER}"
        print(OUTPUT_FOLDER)
        
        # Get sprite URLs from the webpage
        sprite_urls = get_sprite_urls(BASE_URL)
        
        if sprite_urls:
            # Download and process all sprites
            download_pokemon_sprites(BASE_URL, sprite_urls, OUTPUT_FOLDER)
        else:
            print("No sprite URLs found. Please check the website URL and try again.")
