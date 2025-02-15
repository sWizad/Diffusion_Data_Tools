### crop by YOLO World, work best for cartoon
### Version 2 : use pipeline
import os
from lib.pipelines import auto_pipeline
from fastprogress import master_bar, progress_bar


if __name__ == "__main__":
    folder_path =  r'E:\Research\Diffusion_Data_Tools\data\movie\labyrinth'
    base_dir = r"E:\Research\symlink\CivitAI\movie\labyeinth\crop"
    classes = ["face", "person", "animal",]

    pipe = auto_pipeline(classes,base_dir=base_dir,
                         sr_model_path=None
                         )
    pipe.confidence_threshold = 0.2 #defult: 0.2
    pipe.dim_threshold=0.85
    pipe.similarity_threshold=0.95
    pipe.crop_pixels=140
    pipe.pre_process = pipe.crop_image_top_bottom
    #pipe.pre_process = lambda image: pipe.crop_image_top_bottom(image, crop_pixels=24)
    #pipe.pre_process = None
 
    if 0:
        out_dir = pipe.make_folder("") 
        pipe.process_images_in_folder(folder_path, out_dir)
        pipe.delete_similar_image_in_subfolders(out_dir)
        pipe.upsampling_small_images(out_dir)
    else:
        pipe.mb = master_bar(os.listdir(folder_path))
        for subfolder_name in pipe.mb:
            subfolder_path = os.path.join(folder_path, subfolder_name)
            out_dir = pipe.make_folder(subfolder_name)
            pipe.process_images_in_folder(subfolder_path, out_dir)
            pipe.delete_similar_image_in_subfolders(out_dir)
            pipe.upsampling_small_images(out_dir)
    