import os
import random
import re

def remove_brackets(words):
    # Remove text inside parentheses and brackets including the brackets themselves
    words = re.sub(r'\((.*?)\)', r'\1', words)
    words = re.sub(r'\[(.*?)\]', r'\1', words)
    return words

def remove_substrings(words):
    word_list = [word.strip() for word in words.split(',')]
    words_to_remove = set()
    for word in word_list:
        for other_word in word_list:
            if word != other_word and word in other_word:
                words_to_remove.add(word)
    filtered_words = [word for word in word_list if word not in words_to_remove]
    return ', '.join(filtered_words)

def drop_words_with_chance(words, trigger_rate = 0.3, drop_chance=0.6):
    if len(words) == 0:
        return "cartoon"
    
    if random.random() > trigger_rate:
        return words
    
    word_list = words.split(', ')
    filtered_words = [word for word in word_list if random.random() > drop_chance]
    if len(filtered_words)==0:
        return words
    else:
        return ', '.join(filtered_words)

def add_folder_name_to_files(main_folder):
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(root)
                with open(file_path, 'r+') as f:
                    content = f.read()
                    content = remove_brackets(content)
                    
                    if folder_name not in ["0other", "other"]:
                        processed_content = remove_substrings(content)
                        processed_content = drop_words_with_chance(processed_content)
                        content = f"{folder_name}, {processed_content}"
                    
                    f.seek(0)
                    f.write(content)
                    f.truncate()

# Example usage
folder_path = r"D:\Project\CivitAI\Disney\Princess\Mulan\draft"
add_folder_name_to_files(folder_path)

