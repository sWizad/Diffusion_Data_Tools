### crop by YOLO World
import cv2, os, sys
from fastprogress import master_bar, progress_bar
import numpy as np
import torch
import re, random, shutil
try:
    import supervision as sv
except ImportError:
    print("Error: Please install it using: pip install -q supervision==0.19.0rc3")
    sys.exit(1)
try:
    from inference.models.yolo_world.yolo_world import YOLOWorld
except ImportError:
    print("Error: Please install it using: pip install -q inference-gpu[yolo-world], pip install ultralytics, pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)
try:
    import fiftyone as fo
    import fiftyone.zoo as foz
except ImportError:
    print("Error: Please install it using: pip install -q fiftyone ftfy")
    sys.exit(1)
from sklearn.metrics.pairwise import cosine_similarity


def get_YoloModel(model_id, classes):
    model = YOLOWorld(model_id=model_id)
    model.set_classes(classes)
    return model

def get_clipModel(model_id="clip-vit-base32-torch"):
    return foz.load_zoo_model(model_id)

def get_YOLO_and_CLIP_model(classes, yolo_id = "yolo_world/l", clip_id = "clip-vit-base32-torch"):
    model = get_YoloModel(yolo_id, classes)
    clip_model = get_clipModel(clip_id)
    return model, clip_model

def get_SwinIR_model(model_path = 'wd14_tagger_model\003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth', scale = 2):
    from lib.swinir.swinir_network import SwinIR as net

    model = net(upscale=scale, in_chans=3, img_size=64, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
    model.eval()
    return model.to('cuda')

MODEL_NAME = "clip-vit-base32-torch"


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

def scoring_prompt_old(folder_names):
    updated_folder_names = []
    for folder in folder_names:
        updated_folder_names.append(folder)
        if folder == 'score_9':
            updated_folder_names.append('score_9')
            updated_folder_names.append('score_8_up')
            updated_folder_names.append('score_7_up')
        if folder == 'score_8':
            updated_folder_names.append('score_8_up')
            updated_folder_names.append('score_7_up')
        if folder == 'score_7':
            updated_folder_names.append('score_7_up')
        if folder == 'score_6':
            updated_folder_names.append('score_6_up')
            updated_folder_names.append('score_5_up')
            updated_folder_names.append('score_4_up')
        if folder == 'score_5':
            updated_folder_names.append('score_5_up')
            updated_folder_names.append('score_4_up')
        if folder == 'score_4':
            updated_folder_names.append('score_4_up')
    return updated_folder_names

def scoring_prompt(folder_names):
    replacements = {
        'score_9': ['score_7_up', 'score_8_up', 'score_9',],
        'score_8': ['score_7_up', 'score_8_up',],
        'score_7': ['score_6_up', 'score_7_up',],
        'score_6': ['score_4_up', 'score_5_up', 'score_6_up',],
        'score_5': ['score_4_up', 'score_5_up',],
        'score_4': ['score_4_up']
    }
    
    updated_folder_names = []
    for folder in folder_names:
        if folder in replacements:
            updated_folder_names.extend(replacements[folder])
        else:
            updated_folder_names.append(folder)
    
    return updated_folder_names

def add_folder_name_to_files(main_folder, mode="last_only", tag_dropping_rate = 0.3, drop_chance = 0.6):
    for root, dirs, files in os.walk(main_folder):
        image_files = {os.path.splitext(file)[0] for file in files if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))}
        for file in files:
            if file.endswith('.txt'):
                txt_base_name = os.path.splitext(file)[0]
                if txt_base_name in image_files:
                    file_path = os.path.join(root, file)
                    #folder_name = get_folder_name_old(root)
                    folder_names = get_folder_names(root, main_folder)
                    with open(file_path, 'r+') as f:
                        content = f.read()
                        content = remove_brackets(content)
                        
                        #if folder_names[-1] not in ["0other", "other"]:
                        if '0other' not in folder_names:
                            processed_content = remove_substrings(content)
                            processed_content = drop_words_with_chance(processed_content, tag_dropping_rate, drop_chance)
                        else: 
                            processed_content = content
                        if mode == 'last_only':
                            folder_names = [folder_names[-1]]
                        if mode == 'all_score':
                            folder_names = scoring_prompt(folder_names)
                        folder_names = [name for name in folder_names if name not in ["other", "0other","minor"]]
                        folder_name_content = ', '.join(folder_names[::-1])
                        content = f"{folder_name_content}, {processed_content}" if len(folder_names) > 0 else processed_content
                        
                        f.seek(0)
                        f.write(content)
                        f.truncate()

def calculate_similarity_matrix(embeddings, batch_size):
    batch_size = min(embeddings.shape[0], batch_size)
    batch_embeddings = np.array_split(embeddings, batch_size)
    similarity_matrices = []
    max_size_x = max(array.shape[0] for array in batch_embeddings)
    max_size_y = max(array.shape[1] for array in batch_embeddings)
    for batch_embedding in batch_embeddings:
        similarity = cosine_similarity(batch_embedding)
        padded_array = np.zeros((max_size_x, max_size_y))
        padded_array[0:similarity.shape[0], 0:similarity.shape[1]] = similarity
        similarity_matrices.append(padded_array)
    similarity_matrix = np.concatenate(similarity_matrices, axis=0)
    similarity_matrix = similarity_matrix[0:embeddings.shape[0], 0:embeddings.shape[0]]
    similarity_matrix = cosine_similarity(embeddings)
    similarity_matrix -= np.identity(len(similarity_matrix))
    return similarity_matrix

def make_samples(dataset, similarity_matrix, threshold):
    dataset.match(fo.ViewField("max_similarity") > threshold)
    dataset.tags = ["delete", "has_duplicates"]
    id_map = [s.id for s in dataset.select_fields(["id"])]
    samples_to_remove = set()
    samples_to_keep = set()
    for idx, sample in enumerate(dataset):
        if sample.id not in samples_to_remove:
            samples_to_keep.add(sample.id)
            dup_idxs = np.where(similarity_matrix[idx] > threshold)[0]
            for dup in dup_idxs:
                samples_to_remove.add(id_map[dup])
            if len(dup_idxs) > 0:
                sample.tags.append("has_duplicates")
                sample.save()
        else:
            sample.tags.append("delete")
            sample.save()
    return samples_to_remove, samples_to_keep

def delete_similar_images(image_folder, similarity_threshold=0.95, embedding_batch_size=200, similarity_matrix_batch_size=1000, model=None):
    dataset = fo.Dataset.from_dir(dataset_dir=image_folder, dataset_type=fo.types.ImageDirectory)
    
    initial_image_count = len(dataset)
    if model is None: model = foz.load_zoo_model(MODEL_NAME)

    embeddings = dataset.compute_embeddings(model, batch_size=embedding_batch_size)
    similarity_matrix = calculate_similarity_matrix(embeddings, similarity_matrix_batch_size)
    samples_to_remove, samples_to_keep = make_samples(dataset, similarity_matrix, similarity_threshold)

    for sample_id in samples_to_remove:
        sample = dataset[sample_id]
        image_path = sample.filepath
        if os.path.exists(image_path):
            os.remove(image_path)
    
    retained_image_count = len(samples_to_keep)
    deleted_image_count = len(samples_to_remove)
    
    del embeddings, similarity_matrix, samples_to_remove, samples_to_keep
    torch.cuda.empty_cache()

    print(f"Initial image count: {initial_image_count}, Images retained: {retained_image_count}, Images deleted: {deleted_image_count}")

def get_folder_name(dirpath):
    folder_name = os.path.basename(dirpath)

    original_name = re.sub(r'^\d+_', '', folder_name)
    if original_name is None: original_name = folder_name
    return original_name

def get_folder_names(dirpath, main_root = None):
    if main_root is None:
        folder_name = os.path.basename(dirpath)

        original_name = re.sub(r'^\d+_', '', folder_name)
        if original_name is None: original_name = folder_name
        return original_name
    
    relative_path = os.path.relpath(dirpath, main_root)
    subfolders = relative_path.split(os.sep)
    
    cleaned_subfolders = []
    for folder in subfolders:
        cleaned_name = re.sub(r'^\d+_', '', folder)
        cleaned_subfolders.append(cleaned_name)
    
    return cleaned_subfolders

def rename_subfolders(root_folder, num_image_per_epoch = 160):
    for dirpath, dirnames, filenames in os.walk(root_folder, topdown=False):
        image_count = sum(1 for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')))
        if image_count == 0: 
            continue
        
        current_path = dirpath
        original_name = get_folder_name(dirpath)

        if original_name in ["other", "0other"]:
            prefix = 1
        else:
            prefix = max(1, min(4, round(num_image_per_epoch / image_count )))
            
        parent_dir = os.path.dirname(dirpath)
        new_path = os.path.join(parent_dir, f"{prefix}_{original_name}")
        os.rename(current_path, new_path)

def copy_images_to_folder(root_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for dirpath, _, filenames in os.walk(root_folder):
        relative_path = os.path.relpath(dirpath, root_folder)
        path_parts = relative_path.split(os.sep)
        #if len(path_parts) < 2:
        #    continue
        sub_folder = path_parts[0]
        new_sub_folder_path = os.path.join(output_folder, sub_folder)

        if not os.path.exists(new_sub_folder_path):
            os.makedirs(new_sub_folder_path)
            
        for image_file in filenames:
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                continue

            base_name, ext = os.path.splitext(image_file)
            txt_file = f"{base_name}.txt"

            src_image_path = os.path.join(dirpath, image_file)
            dest_image_path = os.path.join(new_sub_folder_path, image_file)

            count = 1
            while os.path.exists(dest_image_path):
                dest_image_path = os.path.join(new_sub_folder_path, f"{base_name}_{count}{ext}")
                count += 1

            shutil.copy2(src_image_path, dest_image_path)

            src_txt_path = os.path.join(dirpath, txt_file)
            if os.path.exists(src_txt_path):
                dest_txt_path = os.path.join(new_sub_folder_path, txt_file)
                count = 1
                while os.path.exists(dest_txt_path):
                    dest_txt_path = os.path.join(new_sub_folder_path, f"{base_name}_{count}.txt")
                    count += 1
                shutil.copy2(src_txt_path, dest_txt_path)

def delete_similar_image_in_subfolders(parent_folder, model=None, similarity_threshold = 0.95):
    if model is None: model = get_clipModel()
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        if os.path.isdir(subfolder_path):
            if not os.listdir(subfolder_path):
                print(f"Skipping empty folder: {subfolder}")
                continue
            print(f"Processing folder: {subfolder}")
            delete_similar_images(subfolder_path, model=model, similarity_threshold=similarity_threshold)

confidence_threshold = 0.05

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return duration, frame_count

def resize_large_img_old(image, max_size = 1024, use_target_ratio= True):
    #if image.shape[0] > 1024 or image.shape[1] > 1024:
    # Calculate aspect ratio
    aspect_ratio = image.shape[1] / image.shape[0]
    if use_target_ratio:
        aspect_ratio = min(TARGET_RATIOS, key=lambda x: abs(x - aspect_ratio))

    # Resize keeping aspect ratio
    if image.shape[0] > image.shape[1]:
        new_height = max_size #// 16
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = max_size #// 16
        new_height = int(new_width / aspect_ratio)
    
    #new_width = new_width  * 16
    #new_height = new_height * 16
        
    image = cv2.resize(image, (new_width, new_height))
    return image

TARGET_SIZES = {'512' :[(512, 512), (448, 576), (576, 448)],
                '1024':[(1024, 1024), (896, 1152), (1152, 896)]}

def resize_large_img(image, max_size = 1024,):
    # Calculate aspect ratio
    aspect_ratio = image.shape[1] / image.shape[0]

    #target_size = min(TARGET_SIZES[str(max_size)], key=lambda size: abs(size[0] / size[1] - aspect_ratio))
            # Find the closest target size based on aspect ratio and dimensions
    def size_difference(size):
        target_aspect_ratio = size[0] / size[1]
        return abs(target_aspect_ratio - aspect_ratio) #+ abs(size[0] - image.shape[0]) + abs(size[1] - image.shape[1])

    target_size = min(TARGET_SIZES[str(max_size)], key=size_difference)

    # Resize image to the target size
    image = cv2.resize(image, target_size)
    return image


def crop_image_by_model_backup(image, model, output_image_path):
    results = model.infer(image,confidence=confidence_threshold)
    detections = sv.Detections.from_inference(results)

    count = 0
    for id, xyxy in zip(detections.class_id,detections.xyxy):
        x1, y1, x2, y2 = map(int, xyxy)
        if abs(x2-x1) < 250 or abs(y2-y1) < 250:
            continue
        cropped_image = image[y1:y2, x1:x2]

        directory, filename = os.path.split(output_image_path)
        cropped_output_path = os.path.join(directory, f"{id:02d}", filename)
        if count > 0:
            cropped_output_path = cropped_output_path.replace('.', f'_{count}.')
        else:
            cropped_output_path = cropped_output_path

        cropped_image = resize_large_img(cropped_image)
        cv2.imwrite(cropped_output_path, cropped_image)
        count = count + 1

    if count == 0 :
        directory, filename = os.path.split(output_image_path)
        cropped_output_path = os.path.join(directory, "0x", filename)
        cv2.imwrite(cropped_output_path, resize_large_img(image))

# Define target ratios
TARGET_RATIOS = [1.0, 3/4, 4/3]
def adjust_crop_coordinates(image, x1, y1, x2, y2):
    height, width = image.shape[:2]
    crop_width, crop_height = x2 - x1, y2 - y1
    
    # Calculate current aspect ratio
    current_ratio = crop_width / crop_height
    
    # Find the closest target ratio
    closest_ratio = min(TARGET_RATIOS, key=lambda x: abs(x - current_ratio))
    
    # Calculate the center of the original crop
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    
    # Try to extend the crop to fit the ratio
    if closest_ratio > current_ratio:
        # Need to increase width
        new_width = (crop_height * closest_ratio)
        new_height = crop_height
    else:
        # Need to increase height
        new_width = crop_width
        new_height = (crop_width / closest_ratio)
    
    # Check if extended crop fits within the image
    if (center_x - new_width/2 >= 0 and center_x + new_width/2 <= width and
        center_y - new_height/2 >= 0 and center_y + new_height/2 <= height):
        # Extended crop fits, use these dimensions
        x1_new = max(0, center_x - new_width / 2)
        x2_new = min(width, x1_new + new_width)
        y1_new = max(0, center_y - new_height / 2)
        y2_new = min(height, y1_new + new_height)
    else:
        # Extended crop doesn't fit, need to shrink
        if closest_ratio > current_ratio:
            # Shrink height
            new_height = (crop_width / closest_ratio)
            new_width = crop_width
        else:
            # Shrink width
            new_width = (crop_height * closest_ratio)
            new_height = crop_height
        
        x1_new = max(0, center_x - new_width / 2)
        x2_new = min(width, x1_new + new_width)
        y1_new = max(0, center_y - new_height / 2)
        y2_new = min(height, y1_new + new_height)
    
    # Ensure we don't exceed image boundaries
    x1_new = max(0, x1_new)
    y1_new = max(0, y1_new)
    x2_new = min(width, x2_new)
    y2_new = min(height, y2_new)
    
    return int(x1_new), int(y1_new), int(x2_new), int(y2_new)

def remove_letterbox(image, threshold_sum=20):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sum the pixel values across each row
    row_sums = np.sum(gray, axis=1)
    
    # Find the first and last non-black rows
    non_black = np.where(row_sums > threshold_sum)[0]
    if len(non_black) == 0:
        return image  # Return original image if it's all below the threshold
    
    top = non_black[0]
    bottom = non_black[-1]
    
    # Crop the image to remove the black bars
    cropped = image[top:bottom+1, :]
    
    return cropped

def crop_image_by_model(image, model, output_image_path, confidence_threshold=0.3):
    image = remove_letterbox(image)
    results = model.infer(image, confidence=confidence_threshold)
    detections = sv.Detections.from_inference(results)

    directory, filename = os.path.split(output_image_path)

    count = 0
    for id, xyxy in zip(detections.class_id, detections.xyxy):
        x1, y1, x2, y2 = map(int, xyxy)
        x1, y1, x2, y2 = adjust_crop_coordinates(image, x1, y1, x2, y2)
        
        if abs(x2-x1) < 250 or abs(y2-y1) < 250:
            continue
        
        cropped_image = image[y1:y2, x1:x2]
        max_dimension = max(abs(x2-x1), abs(y2-y1))
        
        if max_dimension > 768:
            cropped_image = resize_large_img(cropped_image, 1024)
        else:
            cropped_image = resize_large_img(cropped_image, 512)
            #cropped_image = upsampling_by_model(cropped_image,swinir)
            #subfolder = "ss"
        subfolder = f"{id:02d}"
        
        output_filename = f"{filename[:-4]}_{count}.{filename[-3:]}" if count > 0 else filename
        count = count + 1
        cropped_output_path = os.path.join(directory, subfolder, output_filename)
        
        os.makedirs(os.path.dirname(cropped_output_path), exist_ok=True)
        cv2.imwrite(cropped_output_path, cropped_image)

    if count == 0:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = adjust_crop_coordinates(image, 0, 0, width, height)
        
        cropped_image = image[y1:y2, x1:x2]        
        cropped_output_path = os.path.join(directory, "xx", filename)
        cropped_image = resize_large_img(cropped_image, 1024)
        os.makedirs(os.path.dirname(cropped_output_path), exist_ok=True)
        cv2.imwrite(cropped_output_path, cropped_image)

#ASP_RATIOS = [1/1, 1/2, 2/3, 3/4, 4/5, 2/1, 3/2, 5/3, 5/4, 16/9]
resolutions = []
for n in range(8, 33): # 1:1
    resolutions.append((32 * n, 32 * n))
for n in range(12, 25): # 1:2
    resolutions.append((32 * n, 64 * n))
    resolutions.append((64 * n, 32 * n))
for n in range(5, 13): # 2:3
    resolutions.append((64 * n, 96 * n))
    resolutions.append((96 * n, 64 * n))
for n in range(3, 10): # 3:4
    resolutions.append((96 * n, 128 * n))
    resolutions.append((128 * n, 96 * n))
    
def crop_image_by_model_(image, model, output_image_path):
    results = model.infer(image,confidence=confidence_threshold)
    detections = sv.Detections.from_inference(results)

    count = 0
    for id, xyxy in zip(detections.class_id,detections.xyxy):
        x1, y1, x2, y2 = map(int, xyxy)
        
        if abs(x2-x1) < 250 or abs(y2-y1) < 250:
            continue
        
        # Calculate the current aspect ratio and size
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height
        size = width * height

        # Find the closest resolution
        closest_resolution = min(resolutions, key=lambda res: abs(res[0] / res[1] - aspect_ratio) + abs(res[0] * res[1] - size)/1000)

        # Calculate the new dimensions
        new_width = closest_resolution[0]
        new_height = closest_resolution[1]

        # Center adjust
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        new_x1 = max(int(x_center - new_width / 2), 0)
        new_x2 = new_x1 + new_width
        #if new_x2 > image.shape[1]:
        #    new_x2 = image.shape[1]
        #    new_x1 = new_x2 - new_width

        new_y1 = max(int(y_center - new_height / 2), 0)
        new_y2 = new_y1 + new_height
        #if new_y2 > image.shape[0]:
        #    new_y2 = image.shape[0]
        #    new_y1 = new_y2 - new_height

        cropped_image = image[new_y1:new_y2, new_x1:new_x2]

        directory, filename = os.path.split(output_image_path)
        cropped_output_path = os.path.join(directory, f"{id:02d}", filename)
        if count > 0:
            cropped_output_path = cropped_output_path.replace('.', f'_{count}.')
        else:
            cropped_output_path = cropped_output_path

        cropped_image = resize_large_img(cropped_image)
        cv2.imwrite(cropped_output_path, cropped_image)
        count = count + 1

    if count == 0 :
        directory, filename = os.path.split(output_image_path)
        cropped_output_path = os.path.join(directory, "0x", filename)
        cv2.imwrite(cropped_output_path, resize_large_img(image))

def backup_capture_frames(video_path, captured_frames_per_min, out_dir, model, mb=None, idx0 = 0):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    #total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    duration, total_frames = get_video_duration(video_path)
    num_frames = int(duration * captured_frames_per_min / 60)
    frame_indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    frames = []
    print(f"{total_frames} frames from:",video_path)
    
    for idx in progress_bar(frame_indices, parent=mb):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            crop_image_by_model(frame, model, 
                                os.path.join(out_dir, f"f_{1_000_000*idx0 + idx:08}.png"),
                                confidence_threshold=0.3,
                                )
    
    cap.release()
    return frames

def capture_frames(video_path, captured_frames_per_min, out_dir, model, 
                   start_time=0, end_time=None, 
                   mb=None, idx0=0,
                   confidence_threshold = 0.3,
                   ):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    
    duration, total_frames = get_video_duration(video_path)
    
    if end_time is None or end_time > duration:
        end_time = duration

    start_frame = int(start_time * total_frames / duration)
    end_frame = int(end_time * total_frames / duration)
    
    num_frames = int((end_time - start_time) * captured_frames_per_min / 60)
    frame_indices = [int(start_frame + i * (end_frame - start_frame) / (num_frames - 1)) for i in range(num_frames)]
    
    frames = []
    print(f"{total_frames} frames from:", video_path)
    
    for idx in progress_bar(frame_indices, parent=mb):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            crop_image_by_model(frame, model, 
                                os.path.join(out_dir, f"f_{1_000_000*idx0 + idx:08}.png"),
                                confidence_threshold=confidence_threshold,)
    
    cap.release()
    return frames

def process_images_in_folder(folder_path, model, out_dir, mb=None,):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for filename in progress_bar(os.listdir(folder_path), parent=mb):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                output_path = os.path.join(out_dir, f"{filename}")
                crop_image_by_model(image, model, output_path)