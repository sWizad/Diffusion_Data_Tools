import subprocess
from lib.library import delete_similar_image_in_subfolders, add_folder_name_to_files, rename_subfolders

def tagging_image(working_folder):
    command = [
        r"D:\Project\Code\Diffusion_Data_Tools\myenv\Scripts\accelerate.EXE", "launch",
        r"D:\Project\Code\Diffusion_Data_Tools\lib\tag_images_by_wd14_tagger.py",
        "--batch_size", "1",
        "--caption_extension", ".txt",
        "--caption_separator", ",",
        "--character_threshold", "0.5",
        "--max_data_loader_n_workers", "2",
        "--onnx",
        "--recursive",
        "--remove_underscore",
        "--repo_id", "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
        working_folder,
    ]

    subprocess.run(command)

if __name__ == "__main__":
    working_folder = r'D:\Project\CivitAI\Disney\Princess\Test\draft'  

    delete_similar_image_in_subfolders(working_folder)
    tagging_image(working_folder)
    add_folder_name_to_files(working_folder, tag_dropping_rate = 0.3)
    rename_subfolders(working_folder)