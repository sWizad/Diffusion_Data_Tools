### crop by YOLO World, work best for cartoon
import os
from fastprogress import master_bar
from lib.library import get_YOLO_and_CLIP_model, get_SwinIR_model, capture_frames, delete_similar_image_in_subfolders

if __name__ == "__main__":
    video_folder = r'D:\Project\CivitAI\LoL\arcane\video\other'  
    captured_frames_per_min = 60
    base_dir = r'D:\Project\CivitAI\LoL\arcane\crop2\other'
    classes = ["face", "man", "woman"]
    offset_idx = 0

    if not os.path.exists(video_folder):
        print("Video not found:", video_folder)
        exit()
 
    model, clip_model = get_YOLO_and_CLIP_model( classes)
    #swinir = get_SwinIR_model()

    video_files = sorted(os.listdir(video_folder))
    mb = master_bar(video_files)
    for idx, video_file in enumerate(mb):
        print(idx, video_file)
        #if idx <= 3: continue #for skipping
        
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.basename(video_path)
        video_name_without_ext = os.path.splitext(video_name)[0]
        
        # Calculate total number of frames to capture
        out_dir = os.path.join(base_dir, video_name_without_ext)
        for i in range(len(classes)):
            os.makedirs(os.path.join(out_dir, f"{i:02d}"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, f"ss"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "xx"), exist_ok=True)    

        captured_frames = capture_frames(video_path, captured_frames_per_min, 
                                         out_dir, model, #swinir,
                                         mb=mb, idx0 = idx + offset_idx,
                                         confidence_threshold = 0.2,)
        delete_similar_image_in_subfolders(out_dir, model=clip_model)