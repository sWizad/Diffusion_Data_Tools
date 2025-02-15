import os
import re
import json
from fastprogress import progress_bar

def process_files_with_numbers_v3(txt_path = 'pk_body5.txt', json_path=r'old_script\pksp\fusiondex_links2.json'):
    with open(json_path, 'r') as json_file:
        fusion_dict = json.load(json_file)

    for ii in range(401,501):
        iii = str(ii)
        style_text = fusion_dict.get(iii, iii).lower()
                
        with open(txt_path, 'a') as f:
            f.write(f'--prompt "fusion {style_text}_body" \n')

if __name__ == "__main__":

    process_files_with_numbers_v3()