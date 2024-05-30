import os
import random

def get_text_files(directory):
    text_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                text_files.append(os.path.join(root, file))
    return text_files

def _process_directory(directory, output_file):
    for root, dirs, _ in os.walk(directory):
        for dir in dirs:
            sub_dir = os.path.join(root, dir)
            text_files = get_text_files(sub_dir)
            if len(text_files) >= 2:
                selected_files = random.sample(text_files, 2)
                with open(output_file, 'a') as outfile:
                    for txt_file in selected_files:
                        with open(txt_file, 'r') as infile:
                            for line in infile:
                                outfile.write(f'--prompt "score_9, score_8_up, score_8, score_7_up, volumetric lighting, {line.strip()}" --negative_prompt "score_6, score_5, score_4, score_5_up, score_4_up, simple background, blurry,"\n')

def process_directory(directory, max_num_lines, prompt_per_folder=2, out_path="", suffix=""):
    file_counter = 1
    line_counter = 0
    output_file = os.path.join(out_path, f'test_prompt_{file_counter}.txt')

    # Ensure the first output file is created fresh
    open(output_file, 'w').close()

    for root, dirs, _ in os.walk(directory):
        for dir in dirs:
            sub_dir = os.path.join(root, dir)
            text_files = get_text_files(sub_dir)
            if len(text_files) >= prompt_per_folder:
                selected_files = random.sample(text_files, prompt_per_folder)
                for txt_file in selected_files:
                    with open(txt_file, 'r') as infile:
                        for line in infile:
                            if line_counter >= max_num_lines:
                                file_counter += 1
                                line_counter = 0
                                output_file = os.path.join(out_path, f'test_prompt_{file_counter}.txt')
                                open(output_file, 'w').close()  # Ensure the new file is created fresh
                            with open(output_file, 'a') as outfile:
                                prompt = (
                                    f'--prompt "score_9, score_8_up, score_7_up, volumetric lighting, '
                                    f'{line.strip()}, {suffix}" '
                                    '--negative_prompt "score_6, score_5, score_4, score_5_up, score_4_up, '
                                    'simple background, blurry, grayscale"\n'
                                )
                                #prompt = f'{line.strip()}, {suffix}" \n'
                                outfile.write(prompt)
                            line_counter += 1

directory = r'D:\Project\CivitAI\DC\batman\draft'
out_path = r'D:\Project\CivitAI\DC\batman\sampling'
suffix  = "<lora:batman_pony_v1:0.9>"
max_num_lines = 12

os.makedirs(out_path, exist_ok=True)
process_directory(directory, max_num_lines, out_path = out_path, suffix=suffix)
