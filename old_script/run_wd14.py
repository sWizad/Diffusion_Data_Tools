import subprocess

command = [
    #r"D:\Project\Code\Kohya\kohya_ss\venv\Scripts\accelerate.EXE", "launch",
    #r"D:/Project/Code/Kohya/kohya_ss/sd-scripts/finetune/tag_images_by_wd14_tagger.py",
    r"D:\Project\Code\Diffusion_Data_Tools\myenv\Scripts\accelerate.EXE", "launch",
    r"D:\Project\Code\Diffusion_Data_Tools\lib\tag_images_by_wd14_tagger.py",
    "--batch_size", "1",
    "--caption_extension", ".txt",
    "--caption_separator", ",",
    "--character_threshold", "0.5",
    #"--debug",
    #"--frequency_tags",
    "--max_data_loader_n_workers", "2",
    "--onnx",
    "--recursive",
    "--remove_underscore",
    "--repo_id", "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
    r"D:\Project\CivitAI\Disney\Princess\Test\draft"
]

subprocess.run(command)
