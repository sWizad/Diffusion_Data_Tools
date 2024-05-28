### crop by YOLO World
import cv2, os, sys
from fastprogress import progress_bar
try:
    import supervision as sv
except ImportError:
    print("Error: Please install it using: pip install -q supervision==0.19.0rc3")
    sys.exit(1)
try:
    from inference.models.yolo_world.yolo_world import YOLOWorld
except ImportError:
    print("Error: Please install it using: pip install -q inference-gpu[yolo-world]==0.9.18, pip install ultralytics, pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)
try:
    import fiftyone as fo
    import fiftyone.zoo as foz
except ImportError:
    print("Error: Please install it using: pip install -q fiftyone ftfy")
    sys.exit(1)
from old_script.extract_vdo_by_folder import get_video_duration, process_subfolders
confidence_threshold = 0.01

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

def crop_image(image, model, output_image_path):
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
        cv2.imwrite(cropped_output_path, image)

def capture_frames(video_path, captured_frames_per_min, out_dir, model, mb=None, idx0 = 0):
        
    duration = get_video_duration(video_path)  
    num_frames = int(duration * captured_frames_per_min / 60)

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    frames = []
    print(f"{total_frames} frames from:",video_path)
    
    for idx in progress_bar(frame_indices, parent=mb):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            crop_image(frame, model, os.path.join(out_dir, f"f_{1_000_000*idx0 + idx:08}.png"))
    
    cap.release()
    return frames

if __name__ == "__main__":
    #video_path = 'airel.webm'  
    #num_frames_to_capture = 200
    #base_dir = 'img/ariel/'
    video_path = r'D:\Project\CivitAI\kimetsu no yaiba\video\season2.mp4'  
    captured_frames_per_min = 6
    base_dir = r'D:\Project\CivitAI\kimetsu no yaiba\crop'
    classes = [f"face", f"cartoon object", "person",]
    offset_idx = 2

    video_name = os.path.basename(video_path)
    video_name_without_ext = os.path.splitext(video_name)[0]
    if not os.path.exists(video_path):
        print("video not exists:", video_path)
        exit()

    out_dir = os.path.join(base_dir, video_name_without_ext)
    duration = get_video_duration(video_path)  
    num_frames_to_capture = int(duration * captured_frames_per_min / 60)
    for i in range(len(classes)):
        os.makedirs(os.path.join(out_dir,f"{i:02d}"), exist_ok=True)
    os.makedirs(os.path.join(out_dir,f"0x"), exist_ok=True)    

    model = YOLOWorld(model_id="yolo_world/l")
    model.set_classes(classes)


    captured_frames = capture_frames(video_path, num_frames_to_capture, out_dir, model, idx0 = offset_idx)
    
    clip_model = foz.load_zoo_model("clip-vit-base32-torch")
    process_subfolders(out_dir, model=clip_model)