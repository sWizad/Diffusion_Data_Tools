### crop by YOLO World, work best for cartoon
import os
from lib.library import get_YOLO_and_CLIP_model, capture_frames, delete_similar_image_in_subfolders

if __name__ == "__main__":
    video_path = r'D:\Project\CivitAI\Disney\Princess\Test\video\final.mp4'  
    captured_frames_per_min = 20
    base_dir = r'D:\Project\CivitAI\Disney\Princess\Test\crop'
    classes = [f"face", f"cartoon object", "person",]
    offset_idx = 0

    video_name = os.path.basename(video_path)
    video_name_without_ext = os.path.splitext(video_name)[0]
    if not os.path.exists(video_path):
        print("video not exists:", video_path)
        exit()
 
    model, clip_model = get_YOLO_and_CLIP_model( classes)

    out_dir = os.path.join(base_dir, video_name_without_ext)
    for i in range(len(classes)):
        os.makedirs(os.path.join(out_dir,f"{i:02d}"), exist_ok=True)
    os.makedirs(os.path.join(out_dir,f"0x"), exist_ok=True)    


    captured_frames = capture_frames(video_path, captured_frames_per_min, out_dir, model, idx0 = offset_idx)
    
    delete_similar_image_in_subfolders(out_dir, model=clip_model)