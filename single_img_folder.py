### crop by YOLO World, work best for cartoon
import os
from lib.library import get_YOLO_and_CLIP_model, process_images_in_folder, delete_similar_image_in_subfolders
from fastprogress import master_bar, progress_bar

def process_folders_in_folder(base_folder_path, model, out_dir_base):
    mb = master_bar(os.listdir(base_folder_path))
    for subfolder_name in mb:
        subfolder_path = os.path.join(base_folder_path, subfolder_name)

        if os.path.isdir(subfolder_path):
            video_name = os.path.basename(subfolder_path)
            video_name_without_ext = os.path.splitext(video_name)[0]
            out_dir = os.path.join(out_dir_base, video_name_without_ext)
            for i in range(len(classes)):
                os.makedirs(os.path.join(out_dir, f"{i:02d}"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, "0x"), exist_ok=True)
            process_images_in_folder(subfolder_path, model, out_dir)

if __name__ == "__main__":
    folder_path = r"D:\Project\CivitAI\Disney\princess comic\raw"
    base_dir = r"D:\Project\CivitAI\Disney\princess comic\crop"
    classes = [f"comic", f"cartoon object", "person",]
    #classes = [f"cow", f"cartoon object", "titan",]

    model, clip_model = get_YOLO_and_CLIP_model( classes)

    video_name = os.path.basename(folder_path)
    video_name_without_ext = os.path.splitext(video_name)[0]
 

    out_dir = os.path.join(base_dir, video_name_without_ext)
    for i in range(len(classes)):
        os.makedirs(os.path.join(out_dir,f"{i:02d}"), exist_ok=True)
    os.makedirs(os.path.join(out_dir,f"0x"), exist_ok=True)    


    #process_images_in_folder(folder_path, model, out_dir,)
    process_folders_in_folder(folder_path, model, out_dir)
    
    delete_similar_image_in_subfolders(out_dir, model=clip_model)