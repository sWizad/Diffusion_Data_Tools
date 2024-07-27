import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import os
import re
from tqdm import tqdm

# Configuration
api_key = "953552e54b7c70d33a7dbf58b17904ad"
headers = {"Authorization": f"Bearer {api_key}"}
initial_url = "https://civitai.com/api/v1/images"
#download_path = input("Enter the path where you want to save the images: ")
download_path = r'D:\Project\scraper\majicMIX V6'
#min_width = int(input("Enter minimum width of the image: "))
min_width = 600
#min_height = int(input("Enter minimum height of the image: "))
min_height = 600
#max_images = int(input("Enter the maximum number of images you want to download: "))
max_images = 200
nsfw_only = False
#nsfw_only = input("Do you want to download only NSFW images? (yes/no): ").strip().lower() == 'yes'
#if nsfw_only:
#    initial_url += "?nsfw=true"
#initial_url += "?modelId=133005"
initial_url += "?modelId=43331&modelVersionId=94640"
initial_url += "&nsfw=true"
#initial_url += "&modelVersionId=533941"

# Ensure directory exists
if not os.path.exists(download_path):
    os.makedirs(download_path)

# Regex to clean up prompt text
tag_re = re.compile(r'<.*?>')

downloaded_urls = set()
if os.path.exists("downloaded_urls.log"):
    with open("downloaded_urls.log", "r") as log_file:
        downloaded_urls = set(log_file.readlines())

with open("downloaded_urls.log", "a") as log_file:

    next_url = initial_url
    total_saved = 0

    while next_url and total_saved < max_images:
        response = requests.get(next_url, headers=headers)
        response_data = response.json()

        if 'metadata' in response_data and 'nextPage' in response_data['metadata']:
            next_url = response_data['metadata']['nextPage']
        else:
            next_url = None

        if not 'items' in response_data:
            print(response_data)
            exit()

        filtered_images = [image for image in response_data['items'] if 
                           image['stats']['heartCount'] > 5 and 
                           (image['stats']['heartCount'] + image['stats']['likeCount'])/2 > image['stats']['laughCount'] + image['stats']['cryCount'] and 
                           image['meta'] and 'prompt' in image['meta'] and
                           image['width'] >= min_width and 
                           image['height'] >= min_height and
                           image['url'] not in downloaded_urls]
        
        if len(filtered_images) == 0 : break

        for image in tqdm(filtered_images, desc="Saving images and metadata", unit="image"):
            image_url = image['url']
            if image_url in downloaded_urls:
                continue

            # Download image
            try:
                image_response = requests.get(image_url)
                img = Image.open(BytesIO(image_response.content))
                
                # Convert image to RGB if necessary
                if img.mode in ['RGBA', 'P']:
                    img = img.convert('RGB')

                image_id = image['id']

                # Save image
                img_filename = os.path.join(download_path, f"{image_id}.png")
                img.save(img_filename)

                # Save meta.prompt as a text file after cleaning
                meta_prompt = tag_re.sub('', image['meta']['prompt'])
                meta_filename = os.path.join(download_path, f"{image_id}.txt")
                with open(meta_filename, "w", encoding='utf-8') as meta_file:
                    meta_file.write(meta_prompt)

                log_file.write(image_url + "\n")
                downloaded_urls.add(image_url)
                total_saved += 1

                if total_saved >= max_images:
                    break

            except UnidentifiedImageError:
                print(f"Failed to identify the image from URL: {image_url}")

print(f"Downloaded and saved {total_saved} images and metadata files.")