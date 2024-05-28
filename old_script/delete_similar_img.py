import os
import shutil
import numpy as np
import torch
try:
    import fiftyone as fo
    import fiftyone.zoo as foz
except ImportError:
    print("Error: Please install it using: pip install -q fiftyone ftfy")
    sys.exit(1)
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "clip-vit-base32-torch"

def make_embeddings(model, batch_size):
    embeddings = dataset.compute_embeddings(model, batch_size=batch_size)
    del model
    torch.cuda.empty_cache()
    return embeddings

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

def delete_similar_images(image_folder, similarity_threshold=0.98, embedding_batch_size=200, similarity_matrix_batch_size=1000, model=None):
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

def process_subfolders_old_version(parent_folder, model=None):
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Processing folder: {subfolder}")
            delete_similar_images(subfolder_path, model=model)

def is_contains_images(folder):
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
    for file in os.listdir(folder):
        if file.lower().endswith(image_extensions):
            return True
    return False

def process_subfolders(parent_folder, model=None):
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        if is_contains_images(subfolder_path):
            print(f"Processing folder: {subfolder_path}")
            delete_similar_images(subfolder_path, model=model)
        else:
            process_subfolders(subfolder_path, model=model)


model = foz.load_zoo_model("clip-vit-base32-torch")
if 0:
    delete_similar_images(r'D:\Project\CivitAI\DC\batman\draft', model=model)
else:
    process_subfolders(r'D:\Project\CivitAI\Disney\Princess\Mulan\draft', model=model)