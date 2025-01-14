import os, sys, random, json
from lib.library import delete_similar_image_in_subfolders, add_folder_name_to_files, rename_subfolders, copy_images_to_folder, delete_similar_images

def rename_txt_files(directory, suffix='_wd14'):
    for root, _, files in os.walk(directory):
        image_files = {os.path.splitext(file)[0] for file in files if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))}
        for file in files:
            if file.endswith('.txt') and not file.endswith(f'{suffix}.txt'):
                txt_base_name = os.path.splitext(file)[0]
                if txt_base_name in image_files:
                    old_file_path = os.path.join(root, file)
                    new_file_path = os.path.join(root, f"{txt_base_name}{suffix}.txt")
                    if os.path.exists(new_file_path):
                        os.remove(new_file_path)
                    os.rename(old_file_path, new_file_path)

def create_json_files(directory, key = 'prompt_wd14'):
    for root, _, files in os.walk(directory):
        image_files = {os.path.splitext(file)[0] for file in files if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))}
        for file in files:
            if file.endswith('.txt'):
                txt_base_name = os.path.splitext(file)[0]
                if txt_base_name in image_files:
                    txt_file_path = os.path.join(root, file)
                    json_file_path = os.path.join(root, f"{txt_base_name}_meta.json")
                    
                    # Read content from the text file
                    with open(txt_file_path, 'r') as txt_file:
                        content = txt_file.read().strip()
                    
                    # Create a dictionary with the content
                    new_data = {key: content}
                    
                    # Check if JSON file already exists
                    if os.path.exists(json_file_path):
                        # If it exists, load the existing data
                        with open(json_file_path, 'r') as json_file:
                            existing_data = json.load(json_file)
                        # Update the existing data with new data
                        existing_data.update(new_data)
                        data_to_write = existing_data
                    else:
                        data_to_write = new_data
                    
                    # Write the data to the JSON file
                    with open(json_file_path, 'w') as json_file:
                        json.dump(data_to_write, json_file, indent=4)
                    
                    # Optionally, remove the original text file
                    os.remove(txt_file_path)

def delete_flipped_images(directory, suffix='_wd14.txt'):
    # walk through the directory and its subfolders
    for root, dirs, files in os.walk(directory):
        flipped_images = [os.path.join(root, f) for f in files if f.endswith(suffix)]

        # delete the flipped images
        for flipped_image_path in flipped_images:
            os.remove(flipped_image_path)

def combine_and_cleanup(directory):
    for root, _, files in os.walk(directory):
        txt_files = {f.replace('_wd14.txt', '').replace('_blip.txt', '') for f in files if f.endswith('_wd14.txt') or f.endswith('_blip.txt')}
        for base_name in txt_files:
            original_path = os.path.join(root, f"{base_name}.txt")
            wd14_path = os.path.join(root, f"{base_name}_wd14.txt")
            blip_path = os.path.join(root, f"{base_name}_blip.txt")

            contents = []

            if os.path.exists(blip_path):
                with open(blip_path, 'r', encoding='utf-8') as blip_file:
                    contents.append(blip_file.read().replace('\n', ' '))

            if os.path.exists(wd14_path):
                with open(wd14_path, 'r', encoding='utf-8') as wd14_file:
                    contents.append(wd14_file.read().replace('\n', ' '))

            combined_content = ', '.join(contents)
            
            if os.path.exists(original_path): os.remove(original_path)

            with open(original_path, 'w', encoding='utf-8') as combined_file:
                combined_file.write(combined_content)


def process_json_to_txt_v2(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_file_path = os.path.join(root, file)
                txt_file_path = os.path.join(root, file.replace('.json', '.txt'))
                
                # Read content from the JSON file
                with open(json_file_path, 'r') as json_file:
                    data = json.load(json_file)
                
                # Extract prompts
                prompt_cap = data.get('pg_cap', '')
                prompt_tag = data.get('pg_tag', '')
                
                # Combine prompts
                combined_prompt = f"{prompt_cap}, {prompt_tag}".strip()
                
                # Write the combined prompt to a text file
                with open(txt_file_path, 'w') as txt_file:
                    txt_file.write(combined_prompt)

if __name__ == "__main__":
    working_folder = r'E:\Research\symlink\CivitAI\real\lower_p\draft'
    output_folder = r'E:\Research\symlink\CivitAI\anime\dandadan\img' 

    if 0:
        delete_similar_image_in_subfolders(working_folder, similarity_threshold = 0.92)
        #0.92 del 50-60%
    else:
        process_json_to_txt_v2(working_folder)
        add_folder_name_to_files(working_folder, mode='all_score', tag_dropping_rate = 0.5, drop_chance = 0.5)
        copy_images_to_folder(working_folder, output_folder)
        rename_subfolders(output_folder, num_image_per_epoch=200)