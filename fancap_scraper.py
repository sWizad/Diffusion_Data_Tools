import requests
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from io import BytesIO

def get_image_dimensions(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img.size  # (width, height)
    except Exception as e:
        print(f"Error getting dimensions for {url}: {str(e)}")
    return None

def download_image(url, filename, expected_dimensions):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            if img.size == expected_dimensions:
                img.save(filename)
                return True
    except Exception:
        pass
    return False

def download_batch(start_number, total_images, expected_dimensions, out_dir):
    successful_downloads = 0
    consecutive_failures = 0

    def download_task(i):
        nonlocal successful_downloads, consecutive_failures
        url = base_url.format(start_number + i)
        filename = os.path.join(out_dir, f"{start_number + i}.jpg")
        if download_image(url, filename, expected_dimensions):
            successful_downloads += 1
            consecutive_failures = 0
            return True
        consecutive_failures += 1
        return False

    with ThreadPoolExecutor(max_workers=10) as executor:
        for _ in tqdm(executor.map(download_task, range(total_images)), 
                      total=total_images, 
                      desc=f"Downloading images to {out_dir}", 
                      unit="image"):
            if consecutive_failures >= 100:
                print(f"\nStopping early due to 100 consecutive failures in {out_dir}")
                break

    return successful_downloads, consecutive_failures >= 100

#base_url = "https://ancdn.fancaps.net/{}.jpg"
#base_url = "https://cdni.fancaps.net/file/fancaps-movieimages/{}.jpg"
base_url = "https://cdni.fancaps.net/file/fancaps-tvimages/{}.jpg"

def main():
    start_number = 272473
    total_images = 500
    out_dir = 'data/GOT/s3/'
    max_iterations = 30
    idx_offset = 0

    # Get dimensions of the start image
    #start_url = f"https://ancdn.fancaps.net/{start_number}.jpg"
    #start_url = f"https://cdni.fancaps.net/file/fancaps-movieimages/{start_number}.jpg"
    #start_url = f"https://cdni.fancaps.net/file/fancaps-tvimages/{start_number}.jpg"
    start_url = base_url.format(start_number)
    expected_dimensions = get_image_dimensions(start_url)

    if expected_dimensions is None:
        print(f"Failed to get dimensions of the start image: {start_url}")
        return

    print(f"Reference image dimensions: {expected_dimensions[0]}x{expected_dimensions[1]} (width x height)")

    for iteration in range(max_iterations):
        index = idx_offset + iteration
        current_out_dir = os.path.join(out_dir, f'batch_{index}')
        os.makedirs(current_out_dir, exist_ok=True)

        successful_downloads, early_stop = download_batch(start_number, total_images, expected_dimensions, current_out_dir)

        print(f"\nBatch {index}: Successfully downloaded {successful_downloads} out of {total_images} images.")

        if early_stop:
            print(f"Stopping after batch {index} due to too many consecutive failures.")
            break

        start_number += total_images

if __name__ == "__main__":
    main()