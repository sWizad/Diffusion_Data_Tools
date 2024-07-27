import requests
from PIL import Image, ImageOps, UnidentifiedImageError
from io import BytesIO
import os
import re
from tqdm import tqdm

# Configuration
api_key = "953552e54b7c70d33a7dbf58b17904ad"
HEADERS = {"Authorization": f"Bearer {api_key}"}
initial_url = "https://civitai.com/api/v1/images"
MIN_WIDTH = 600
MIN_HEIGHT = 600
MAX_IMAGES = 200
DOWNLOADED_URLS_LOG = "downloaded_urls.log"

DOWNLOAD_PATH = r'D:\Project\scraper\Juggernaut V8'
initial_url += "?modelId=133005&modelVersionId=288982"
#initial_url += "&nsfw=true"
#initial_url += "&modelVersionId=533941"

if not os.path.exists(DOWNLOAD_PATH):
    os.makedirs(DOWNLOAD_PATH)

# Regex to clean up prompt text
tag_re = re.compile(r'<.*?>')


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_downloaded_urls():
    if os.path.exists(DOWNLOADED_URLS_LOG):
        with open(DOWNLOADED_URLS_LOG, "r") as log_file:
            return set(log_file.readlines())
    else:
        return set()

def save_downloaded_url(url, log_file):
    log_file.write(url + "\n")

def filter_images(images):
    return [image for image in images if
            image['stats']['heartCount'] > 5 and
            (image['stats']['heartCount'] + image['stats']['likeCount'])/2 > image['stats']['laughCount'] + image['stats']['cryCount'] and
            image['meta'] and 'prompt' in image['meta'] and
            image['width'] >= MIN_WIDTH and
            image['height'] >= MIN_HEIGHT]

aspect_ratios = {
    "640x960": 2 / 3,
    "768x1024": 3 / 4,
    "960x640": 3 / 2,
    "1024x1024": 1,
    "1024x768": 4 / 3,
}

def resize_image(img, closest_aspect_ratio):
    width, height = map(int, closest_aspect_ratio.split("x"))
    size = (width, height)

    # Crop image to the closest aspect ratio and resize to the desired size
    crop_size = (img.width, int(img.width / aspect_ratios[closest_aspect_ratio]))
    img = ImageOps.fit(img, crop_size, centering=(0.5, 0.5))
    img = img.resize(size)

    return img

def download_and_save_image(image, downloaded_urls, log_file, total_saved, download_path):
    image_url = image['url']
    if image_url in downloaded_urls:
        return

    try:
        image_response = requests.get(image_url)
        img = Image.open(BytesIO(image_response.content))

        # Convert image to RGB if necessary
        if img.mode in ['RGBA', 'P']:
            img = img.convert('RGB')

        # Determine the closest aspect ratio
        img_aspect_ratio = img.width / img.height
        closest_aspect_ratio = min(aspect_ratios, key=lambda x: abs(aspect_ratios[x] - img_aspect_ratio))

        # Resize image
        img = resize_image(img, closest_aspect_ratio)

        image_id = image['id']

        # Save image
        img_filename = os.path.join(download_path, f"{image_id}.png")
        img.save(img_filename)

        # Save meta.prompt as a text file after cleaning
        meta_prompt = tag_re.sub('', image['meta']['prompt'])
        meta_filename = os.path.join(download_path, f"{image_id}.txt")
        with open(meta_filename, "w", encoding='utf-8') as meta_file:
            meta_file.write(meta_prompt)

        save_downloaded_url(image_url, log_file)
        downloaded_urls.add(image_url)
        total_saved += 1

        return total_saved
    
    except UnidentifiedImageError:
        print(f"Failed to identify the image from URL: {image_url}")
        return total_saved

def download_save(initial_url, max_images = 200, download_path = DOWNLOAD_PATH):
    downloaded_urls = load_downloaded_urls()

    with open(DOWNLOADED_URLS_LOG, "a") as log_file:
        next_url = initial_url
        total_saved = 0

        while next_url and total_saved < max_images:
            response = requests.get(next_url, headers=HEADERS)
            response_data = response.json()

            if 'metadata' in response_data and 'nextPage' in response_data['metadata']:
                next_url = response_data['metadata']['nextPage']
            else:
                next_url = None

            if not 'items' in response_data:
                print(response_data)
                exit()

            filtered_images = filter_images(response_data['items'])

            if len(filtered_images) == 0:
                break

            for image in tqdm(filtered_images, desc="Saving images and metadata", unit="image"):
                total_saved = download_and_save_image(image, downloaded_urls, log_file, total_saved, download_path)

                if total_saved >= max_images:
                    break

def main_old():
    download_path = r'D:\Project\scraper\Juggernaut V8'
    create_dir_if_not_exists(download_path)

    initial_url = "https://civitai.com/api/v1/images"
    initial_url += "?modelId=133005&modelVersionId=288982"

    download_save(initial_url,max_images=200 , download_path=download_path)
    download_save(initial_url+"&nsfw=true", max_images=50, download_path=download_path)

def main():
    model_list = {
        #'Realistic Vision V5-1': "?modelId=4201&modelVersionId=130072",
        'epiCRealism': "?modelId=25694&modelVersionId=143906",
        'ChilloutMix': "?modelId=6424&modelVersionId=11745",
        'CyberRealistic': "?modelId=15003&modelVersionId=372799",
        'AbsoluteReality': "?modelId=81458&modelVersionId=132760",
        'RealVisXL': "?modelId=139562&modelVersionId=344487",
    }

    for model in model_list:
        download_path = f'D:\Project\scraper\{model}'
        create_dir_if_not_exists(download_path)
        
        initial_url = "https://civitai.com/api/v1/images"
        initial_url += model_list[model] 

        download_save(initial_url,max_images=150 , download_path=download_path)
        download_save(initial_url+"&nsfw=true", max_images=100, download_path=download_path)

if __name__ == "__main__":
    main()
