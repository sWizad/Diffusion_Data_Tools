import os
import random
import zipfile
from collections import defaultdict
from pathlib import Path

def split_and_zip_files(input_folder):
    """
    Split paired PNG and TXT files into 5 random groups and create zip files.
    
    Args:
        input_folder (str): Path to the input folder containing PNG and TXT files
    """
    # Get all PNG files
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    
    # Create pairs dictionary
    file_pairs = {}
    for png_file in png_files:
        base_name = os.path.splitext(png_file)[0]
        txt_file = base_name + '.txt'
        
        # Check if corresponding TXT file exists
        if os.path.exists(os.path.join(input_folder, txt_file)):
            file_pairs[base_name] = (png_file, txt_file)
    
    # Randomly assign pairs to 5 groups
    base_names = list(file_pairs.keys())
    random.shuffle(base_names)
    
    # Calculate items per group
    items_per_group = len(base_names) // 5
    remainder = len(base_names) % 5
    
    # Create groups
    groups = defaultdict(list)
    current_position = 0
    
    for group_num in range(5):
        # Calculate number of items for this group
        items_in_group = items_per_group + (1 if group_num < remainder else 0)
        
        # Add items to group
        group_files = base_names[current_position:current_position + items_in_group]
        for base_name in group_files:
            groups[group_num].append(file_pairs[base_name])
        
        current_position += items_in_group
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(input_folder, 'split_groups')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create zip files for each group
    for group_num, file_list in groups.items():
        zip_path = os.path.join(output_dir, f'0g_all_{group_num + 1}.zip')
        
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            for png_file, txt_file in file_list:
                # Add PNG file to zip
                png_path = os.path.join(input_folder, png_file)
                zip_file.write(png_path, png_file)
                
                # Add TXT file to zip
                txt_path = os.path.join(input_folder, txt_file)
                zip_file.write(txt_path, txt_file)
    
    # Return summary
    return {i: len(files) for i, files in groups.items()}

# Example usage
if __name__ == "__main__":
    # Replace with your input folder path
    input_folder = r"G:\Teng\Full Sprite pack 1-108 (November 2024)\OutForAll"
    
    try:
        group_sizes = split_and_zip_files(input_folder)
        print("\nFiles have been split into groups with the following distribution:")
        for group_num, size in group_sizes.items():
            print(f"Group {group_num + 1}: {size} pairs")
        print("\nZip files have been created in the 'split_groups' subdirectory")
    except Exception as e:
        print(f"An error occurred: {str(e)}")