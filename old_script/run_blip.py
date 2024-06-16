import subprocess, sys

def run_command():
    blip_path = r'D:\Project\Code\Diffusion_Data_Tools\blip'
    if blip_path not in sys.path:
        sys.path.append(blip_path)

    command = [
        #r'D:\Project\Code\Kohya\kohya_ss\venv\Scripts\python.exe',
        #r'D:/Project/Code/Kohya/kohya_ss/sd-scripts/finetune/make_captions.py',
        r"D:\Project\Code\Diffusion_Data_Tools\myenv\Scripts\python.exe",
        r'D:\Project\Code\Diffusion_Data_Tools\lib\make_captions.py',
        '--batch_size', '1',
        '--num_beams', '1',
        '--top_p', '0.9',
        '--max_length', '75',
        '--min_length', '5',
        '--beam_search',
        "--recursive",
        '--caption_extension', '.txt',
        r'D:\Project\CivitAI\Disney\Princess\Test\draft',
        '--caption_weights', 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'
    ]
    
    subprocess.run(command)
    #print(result.stdout)
    #print(result.stderr)

# Example usage
run_command()
