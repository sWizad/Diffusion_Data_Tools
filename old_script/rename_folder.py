import os
import re
#import math

def rename_subfolders(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder, topdown=False):
        image_count = sum(1 for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')))
        if image_count == 0: 
            continue
        prefix = max(1, min(4, round(160 / image_count )))
        folder_name = os.path.relpath(dirpath, root_folder)
        #for dirname in dirnames:
        current_path = dirpath
        if is_prefixed(folder_name):
            original_name = re.sub(r'^\d+_', '', folder_name)
        else:
            original_name = folder_name
            
        new_path = os.path.join(root_folder, f"{prefix}_{original_name}")
        os.rename(current_path, new_path)

def is_prefixed(dirname):
    return re.match(r'^\d+_.+', dirname) is not None

def _rename_subfolders(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder, topdown=False):
        for dirname in dirnames:
            current_path = os.path.join(dirpath, dirname)
            new_path = os.path.join(dirpath, f"4_{dirname}")
            os.rename(current_path, new_path)

if __name__ == "__main__":
    """
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)
    
    root_folder = sys.argv[1]

    if not os.path.isdir(root_folder):
        print(f"Error: {root_folder} is not a valid directory")
        sys.exit(1)
    """
    root_folder = r"D:\Project\CivitAI\DC\batman\draft"
    rename_subfolders(root_folder)
