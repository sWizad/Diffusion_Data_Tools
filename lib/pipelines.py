import cv2, os, sys
from fastprogress import progress_bar
import numpy as np
import torch
from PIL import Image
from .library import get_video_duration, adjust_crop_coordinates, resize_large_img, remove_letterbox, calculate_similarity_matrix, make_samples
try:
    from inference.models.yolo_world.yolo_world import YOLOWorld
except ImportError:
    print("Error: Please install it using: pip install -q inference-gpu[yolo-world], pip install ultralytics, pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)
try:
    import supervision as sv
except ImportError:
    print("Error: Please install it using: pip install -q supervision==0.19.0rc3")
    sys.exit(1)
try:
    import fiftyone as fo
    import fiftyone.zoo as foz
except ImportError:
    print("Error: Please install it using: pip install -q fiftyone ftfy")
    sys.exit(1)

class clip_pipeline():
    def __init__(self, model_id="clip-vit-base32-torch") -> None:
        self.get_clipModel(model_id)

    def get_clipModel(self, model_id="clip-vit-base32-torch"):
        self.clip_model =  foz.load_zoo_model(model_id)
    
    def delete_similar_image_in_subfolders(self, parent_folder):
        for subfolder in os.listdir(parent_folder):
            subfolder_path = os.path.join(parent_folder, subfolder)
            if os.path.isdir(subfolder_path):
                if not os.listdir(subfolder_path):
                    print(f"Skipping empty folder: {subfolder}")
                    continue
                print(f"Processing folder: {subfolder}")
                self.delete_similar_images(subfolder_path)
    
    def delete_similar_images(self, image_folder,  embedding_batch_size=200, similarity_matrix_batch_size=1000):
        dataset = fo.Dataset.from_dir(dataset_dir=image_folder, dataset_type=fo.types.ImageDirectory)
        
        initial_image_count = len(dataset)

        embeddings = dataset.compute_embeddings(self.clip_model, batch_size=embedding_batch_size)
        similarity_matrix = calculate_similarity_matrix(embeddings, similarity_matrix_batch_size)
        samples_to_remove, samples_to_keep = make_samples(dataset, similarity_matrix, self.similarity_threshold)

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

class cropping_pipeline():
    def __init__(self,  classes, model_id="yolo_world/l", base_dir = '', captured_frames_per_min = 40) -> None:
        self.get_YoloModel(model_id, classes)
        self.base_dir = base_dir
        self.classes = classes
        self.mb = None
        # maybe move to different function
        self.captured_frames_per_min = captured_frames_per_min 
        
    def get_YoloModel(self, model_id, classes):
        model = YOLOWorld(model_id=model_id)
        model.set_classes(classes)
        self.model = model
    
    def make_folder(self, video_name_without_ext):
        out_dir = os.path.join(self.base_dir, video_name_without_ext)
        for i in range(len(self.classes)):
            os.makedirs(os.path.join(out_dir, f"{i:02d}"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "xx"), exist_ok=True) 

        return out_dir

    def process_images_in_folder(self,folder_path, out_dir,):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        for filename in progress_bar(os.listdir(folder_path), parent=self.mb):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    output_path = os.path.join(out_dir, f"{filename}")
                    self.crop_image_by_model(image, output_path)

    def capture_frames(self, video_path, out_dir,  
                    start_time=0, end_time=None, 
                    idx0=0,
                    #confidence_threshold = 0.3,
                    #dim_threshold = 0.5,
                    ):
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        
        duration, total_frames = get_video_duration(video_path)
        
        if end_time is None or end_time > duration:
            end_time = duration

        start_frame = int(start_time * total_frames / duration)
        end_frame = int(end_time * total_frames / duration)
        
        num_frames = int((end_time - start_time) * self.captured_frames_per_min / 60)
        frame_indices = [int(start_frame + i * (end_frame - start_frame) / (num_frames - 1)) for i in range(num_frames)]
        
        frames = []
        print(f"{total_frames} frames from:", video_path)
        
        for idx in progress_bar(frame_indices, parent=self.mb):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                self.crop_image_by_model(frame, 
                                    os.path.join(out_dir, f"f_{1_000_000*idx0 + idx:08}.png"),)
        
        cap.release()
        return frames
    
    def crop_image_by_model(self,image, output_image_path,):
        image = remove_letterbox(image)
        results = self.model.infer(image, confidence=self.confidence_threshold)
        detections = sv.Detections.from_inference(results)

        directory, filename = os.path.split(output_image_path)

        count = 0
        for id, xyxy in zip(detections.class_id, detections.xyxy):
            x1, y1, x2, y2 = map(int, xyxy)
            x1, y1, x2, y2 = adjust_crop_coordinates(image, x1, y1, x2, y2)
            
            max_dimension = max(abs(x2-x1), abs(y2-y1))
            if max_dimension < self.dim_threshold * 512:
                continue
            
            cropped_image = image[y1:y2, x1:x2]
            
            if max_dimension >  self.dim_threshold * 1024:
                cropped_image = resize_large_img(cropped_image, 1024)
            else:
                cropped_image = resize_large_img(cropped_image, 512)
            subfolder = f"{id:02d}"
            
            output_filename = f"{filename[:-4]}_{count}.{filename[-3:]}" if count > 0 else filename
            count = count + 1
            cropped_output_path = os.path.join(directory, subfolder, output_filename)
            
            os.makedirs(os.path.dirname(cropped_output_path), exist_ok=True)
            #cv2.imwrite(cropped_output_path, cropped_image)
            save_image(cropped_output_path, cropped_image)

        if count == 0:
            height, width = image.shape[:2]
            x1, y1, x2, y2 = adjust_crop_coordinates(image, 0, 0, width, height)
            
            cropped_image = image[y1:y2, x1:x2]        
            cropped_output_path = os.path.join(directory, "xx", filename)
            cropped_image = resize_large_img(cropped_image, 1024)
            os.makedirs(os.path.dirname(cropped_output_path), exist_ok=True)
            #cv2.imwrite(cropped_output_path, cropped_image)
            save_image(cropped_output_path, cropped_image)

class sr_pipeline():
    def __init__(self, model_path = None):
        self.get_SwinIR_model(model_path=model_path)
        torch.cuda.empty_cache()

    def get_SwinIR_model(self, model_path = None, scale = 2):
        from lib.swinir.swinir_network import SwinIR as net
        if model_path is None: model_path = r'wd14_tagger_model\003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth'

        sr_model = net(upscale=scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        
        param_key_g = 'params_ema'

        pretrained_model = torch.load(model_path)
        sr_model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

        sr_model.eval()
        sr_model.to('cuda')
        self.sr_model = sr_model

    def upsampling(self, img_lq, tile = None, scale = 2, window_size = 8):
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to('cuda')  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = self.sr_model(img_lq)
            output = output[..., :h_old * scale, :w_old * scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        return output
    
    def get_small_images(self, folder_path, size_threshold=512):
        small_images = []
        
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    file_path = os.path.join(root, filename)
                    try:
                        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                        if img is not None:
                            height, width = img.shape[:2]
                            if max(width,height) <= size_threshold:
                                small_images.append(file_path)
                    except Exception as e:
                        print(f"Error opening {file_path}: {str(e)}")
        
        return small_images

    def upsampling_small_images(self, folder_path, size_threshold=512):
        small_images = self.get_small_images(folder_path, size_threshold)
        count = 0
        for file_path in progress_bar(small_images, parent=self.mb):
            # Read image using cv2
            img_lq = cv2.imread(file_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            
            upsampled_image = self.upsampling(img_lq)

            save_image(file_path, upsampled_image)
            count += 1
                    
        print(f"Processed and saved {count} images")


class auto_pipeline(clip_pipeline,cropping_pipeline,sr_pipeline):
    def __init__(self, classes, 
                 yolo_model_id="yolo_world/l", 
                 clip_model_id="clip-vit-base32-torch",
                 sr_model_path=r'wd14_tagger_model\003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth',
                 base_dir = '') -> None:
        self.get_clipModel(clip_model_id)
        self.get_YoloModel(yolo_model_id, classes)
        if sr_model_path is None:
            self.sr_model = None
        else:
            self.get_SwinIR_model(model_path=sr_model_path)
        self.base_dir = base_dir
        self.classes = classes
        self.mb = None

    def upsampling_small_images(self, out_dir):
        if self.sr_model is None: pass
        else: super().upsampling_small_images( out_dir)

def save_image(file_path, image):
    # Extract the file extension
    file_name, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    # Check the file extension and save the image accordingly
    if file_extension == '.jpg' or file_extension == '.jpeg':
        # Save as JPEG with lower quality
        cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, 50])  # Quality range: 0-100
    elif file_extension == '.png':
        new_file_path = file_name + '.png'
        cv2.imwrite(new_file_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])  