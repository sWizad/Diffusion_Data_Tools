### crop by YOLO World
import cv2, os, sys
from fastprogress import master_bar, progress_bar
import numpy as np
import torch
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

MODEL_NAME = "clip-vit-base32-torch"


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

def process_subfolders(parent_folder, model=None):
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        if os.path.isdir(subfolder_path):
            if not os.listdir(subfolder_path):
                print(f"Skipping empty folder: {subfolder}")
                continue
            print(f"Processing folder: {subfolder}")
            delete_similar_images(subfolder_path, model=model)

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
    return duration

def resize_large_img(image):
    if image.shape[0] > 1024 or image.shape[1] > 1024:
        # Calculate aspect ratio
        aspect_ratio = image.shape[1] / image.shape[0]

        # Resize keeping aspect ratio
        if image.shape[0] > image.shape[1]:
            new_height = 1024
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = 1024
            new_height = int(new_width / aspect_ratio)

        image = cv2.resize(image, (new_width, new_height))
    return image

def crop_image_by_model(image, model, output_image_path):
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

def capture_frames(video_path, num_frames,out_dir, model, mb, idx0 = 0):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    frames = []
    print(f"{total_frames} frames from:",video_path)
    
    for idx in progress_bar(frame_indices, parent=mb):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            crop_image_by_model(frame, model, os.path.join(out_dir, f"f_{1_000_000*idx0 + idx:08}.png"))
    
    cap.release()
    return frames

if __name__ == "__main__":
    video_folder = r'D:\Project\CivitAI\Disney\Princess\Test\video'  
    captured_frames_per_min = 40
    base_dir = r'D:\Project\CivitAI\Disney\Princess\Test\crop'
    classes = ["face", "cartoon object", "person"]
    offset_idx = 0

    if not os.path.exists(video_folder):
        print("Video not found:", video_folder)
        exit()
 
    model = YOLOWorld(model_id="yolo_world/l")
    model.set_classes(classes)

    clip_model = foz.load_zoo_model("clip-vit-base32-torch")

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
        duration = get_video_duration(video_path)  
        num_frames_to_capture = int(duration * captured_frames_per_min / 60)

        out_dir = os.path.join(base_dir, video_name_without_ext)
        for i in range(len(classes)):
            os.makedirs(os.path.join(out_dir, f"{i:02d}"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "0x"), exist_ok=True)    


        captured_frames = capture_frames(video_path, num_frames_to_capture, out_dir, model, mb, idx + offset_idx)
        process_subfolders(out_dir, model=clip_model)
