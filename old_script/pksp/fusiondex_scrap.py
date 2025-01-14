import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
from PIL import Image
import io
from fastprogress.fastprogress import master_bar, progress_bar
from multiprocessing import Pool, cpu_count

def process_image(img_data):
    # Create image from bytes
    img = Image.open(io.BytesIO(img_data))
    img = img.convert('RGBA')
    
    bg = Image.new('RGBA', img.size, 'white')
    if img.mode == 'RGBA':
        bg.paste(img, mask=img.split()[3])
    else:
        bg.paste(img)
    
    # Resize down and up
    small = bg.resize((96, 96), Image.Resampling.NEAREST)
    final = small.resize((768, 768), Image.Resampling.NEAREST)
    
    # Convert to RGB
    return final.convert('RGBA')

def download_sprite_images(url, target_folder="data/pkspif"):
    """
    Scrapes a webpage for sprite images and downloads them.
    """
    os.makedirs(target_folder, exist_ok=True)
    successful_downloads = []
    failed_downloads = []
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        sprite_classes = ["sprite-preview sprite-variant-main", "sprite-preview sprite-variant-alt"]
        
        for class_name in sprite_classes:
            # Find articles with the sprite preview class
            articles = soup.find_all('article', class_=class_name)
            
            for article in articles:
                try:
                    # Find the img tag within the article
                    img = article.find('img')
                    if not img:
                        continue
                        
                    img_url = img.get('src')
                    if not img_url:
                        continue
                    
                    # Get sprite ID for filename
                    sprite_id = article.find('span', class_='sprite-id')
                    sprite_id_text = sprite_id.text.strip('#') if sprite_id else 'unknown'
                    sprite_id_text = sprite_id_text.replace(".","_")
                    
                   # Download and process image
                    img_response = requests.get(img_url)
                    img_response.raise_for_status()
                    
                    processed_img = process_image(img_response.content)
                    
                    filename = os.path.join(target_folder, f"{sprite_id_text}.png")
                    processed_img.save(filename, 'PNG')
                    
                    successful_downloads.append(filename)
                    
                except Exception as e:
                    failed_downloads.append((img_url, str(e)))
    
    except Exception as e:
        print(f"Error accessing webpage: {str(e)}")
        return [], [(url, str(e))]
    
    return successful_downloads, failed_downloads

def generate_urls(dex_list):
    """Generate all URLs based on dex_list."""
    urls = []
    for i in dex_list:
        for j in dex_list:
            url = f"https://www.fusiondex.org/{i}.{j}/"
            urls.append(url)
    return urls

def generate_urls2(dex_list):
    """Generate all URLs based on dex_list."""
    urls = []
    for i in dex_list:
        url = f"https://www.fusiondex.org/{i}/"
        urls.append(url)
    return urls

if __name__ == "__main__":
    dex_list = [100,250,290,294,339,392,
                403, 50, 25, 43,201,  1,
                421,151,170,373, 15,133,
                241,468,132,439,406,  6,
                27, 196,  4,358,287, 12,
                3,  370, 26,282,300,360,]
    dex_list = [i for i in range(1,470)]

    urls = generate_urls2(dex_list)
    
    # Use multiprocessing Pool for parallel processing
    num_workers = min(cpu_count(), len(urls))  # Limit workers to number of URLs or CPUs
    with Pool(num_workers) as pool:
        results = list(progress_bar(pool.imap(download_sprite_images, urls), total=len(urls)))
    