import os
import re
import json
from fastprogress import progress_bar

def remove_brackets(words):
    # Remove text inside parentheses and brackets including the brackets themselves
    words = re.sub(r'\((.*?)\)', r'\1', words)
    words = re.sub(r'\[(.*?)\]', r'\1', words)
    return words

def process_files_with_numbers(main_folder,json_path=r'old_script\pksp\fusiondex_links.json'):

    with open(json_path, 'r') as json_file:
        fusion_dict = json.load(json_file)

    for root, dirs, files in os.walk(main_folder):
        image_files = {os.path.splitext(file)[0] for file in files if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))}
        
        for file in files:
            if file.endswith('.txt'):
                txt_base_name = os.path.splitext(file)[0]
                if txt_base_name in image_files:
                    # Get corresponding image filename
                    image_filename = next(f for f in files if os.path.splitext(f)[0] == txt_base_name and f.endswith('.png'))
                    
                    # Extract numbers from filename using regex
                    match = re.match(r'(\d+)[a-zA-Z]?\.png$', image_filename)
                    if match:
                        number1 = match.groups()[0]

                        style_text = fusion_dict.get(number1, number1).lower()  
                        
                        # Write to text file
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r+') as f:
                            content = f.read()
                            f.seek(0)
                            content = remove_brackets(content)

                            new_content = f"{style_text}, {content}"
                            f.write(new_content)
                        
                            f.truncate()

def process_files_with_numbers_v2(main_folder,json_path=r'old_script\pksp\fusiondex_links.json'):

    with open(json_path, 'r') as json_file:
        fusion_dict = json.load(json_file)

    for root, dirs, files in os.walk(main_folder):
        image_files = {os.path.splitext(file)[0] for file in files if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))}
        
        for file in files:
            if file.endswith('.txt'):
                txt_base_name = os.path.splitext(file)[0]
                if txt_base_name in image_files:
                    # Get corresponding image filename
                    image_filename = next(f for f in files if os.path.splitext(f)[0] == txt_base_name and f.endswith('.png'))
                    
                    # Extract numbers from filename using regex
                    match = re.match(r'(\d+)_(\d+)[a-zA-Z]?\.png$', image_filename)
                    if match:
                        number1, number2 = match.groups()

                        style_text = fusion_dict.get(number1, number1).lower()  # Use number as fallback if not found
                        body_text = fusion_dict.get(number2, number2).lower()   # Use number as fallback if not found
                        
                        # Write to text file
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r+') as f:
                            content = f.read()
                            f.seek(0)
                            content = remove_brackets(content)

                            new_content = f"fusion {body_text}_body in {style_text}_style, {content}"
                            f.write(new_content)
                        
                            f.truncate()

def process_files_with_files_name(folder_path):
    for root, _, files in os.walk(folder_path):
        images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        for image_file in progress_bar(images):
            # Image name without extension
            base_name = os.path.splitext(image_file)[0]
            
            # Path to the corresponding .txt file
            txt_file_path = os.path.join(root, base_name + '.txt')

            if os.path.exists(txt_file_path):
                with open(txt_file_path, 'r+') as txt_file:
                    old_content = txt_file.read().strip()

                    # Modify the image name as described
                    modified_name = ', '.join(base_name.split('-')[::-1])

                    # Combine modified name with the old content
                    new_content = modified_name + ', ' + old_content

                    # Rewind and overwrite the content
                    txt_file.seek(0)
                    txt_file.write(new_content)
                    txt_file.truncate()

if __name__ == "__main__":
    folder_path =  r'E:\Research\symlink\CivitAI\pksp\smogon\draft'

    process_files_with_files_name(folder_path)