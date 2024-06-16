### crop by YOLO World, work best for cartoon
import os
from fastprogress import master_bar
from lib.library import get_YOLO_and_CLIP_model, capture_frames, delete_similar_image_in_subfolders

if __name__ == "__main__":
    video_folder = r'D:\Project\CivitAI\Disney\SleepingB\video'  
    captured_frames_per_min = 40
    base_dir = r'D:\Project\CivitAI\Disney\SleepingB\crop'
    classes = ["face", "cartoon object", "person"]
    offset_idx = 0

    if not os.path.exists(video_folder):
        print("Video not found:", video_folder)
        exit()
 
    model, clip_model = get_YOLO_and_CLIP_model( classes)

    video_files = sorted(os.listdir(video_folder))
    mb = master_bar(video_files)
    for idx, video_file in enumerate(mb):
        print(idx, video_file)
        #if idx <= 8: continue
        
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.basename(video_path)
        video_name_without_ext = os.path.splitext(video_name)[0]
        
        # Calculate total number of frames to capture
        out_dir = os.path.join(base_dir, video_name_without_ext)

        out_dir = os.path.join(base_dir, video_name_without_ext)
        for i in range(len(classes)):
            os.makedirs(os.path.join(out_dir, f"{i:02d}"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "0x"), exist_ok=True)    

        captured_frames = capture_frames(video_path, captured_frames_per_min, out_dir, model, mb, idx + offset_idx)
        delete_similar_image_in_subfolders(out_dir, model=clip_model)