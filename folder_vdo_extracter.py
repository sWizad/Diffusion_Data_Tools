### crop by YOLO World, work best for cartoon
### Version 2 : use pipeline
import os
from fastprogress import master_bar
from lib.pipelines import auto_pipeline

if __name__ == "__main__":
    video_folder = r'D:\Project\CivitAI\anime\chainsaw man\video'  
    base_dir = r'D:\Project\CivitAI\anime\chainsaw man\crop'
    classes = ["face", "person", "character"]
    offset_idx = 0
    
    pipe = auto_pipeline(classes,base_dir=base_dir,sr_model_path=None)
    pipe.captured_frames_per_min = 60
    pipe.confidence_threshold = 0.1
    pipe.dim_threshold=0.7
    pipe.similarity_threshold=0.95
 
    if not os.path.exists(video_folder):
        print("Video not found:", video_folder)
        exit()
        
    video_files = sorted(os.listdir(video_folder))
    pipe.mb = master_bar(video_files)
    for idx, video_file in enumerate(pipe.mb):
        print(idx, video_file)
        if idx not in [5, 6, 7]: continue

        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.basename(video_path)
        video_name_without_ext = os.path.splitext(video_name)[0]
        
        out_dir = pipe.make_folder(video_name_without_ext)   

        pipe.capture_frames(video_path, out_dir,
                                        idx0 = idx + offset_idx,)
        pipe.delete_similar_image_in_subfolders(out_dir)
        pipe.upsampling_small_images(out_dir)
    

