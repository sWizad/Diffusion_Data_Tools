import os
import time
import cv2
import torch
import numpy as np
from fastprogress import progress_bar
from lib.RRDBNet_arch import RRDBNet, RealESRGANer
from diffusers import AutoencoderKL
from typing import Tuple
import random

def setup_models(
        model_path = r'E:\Research\stable-diffusion-webui-reForge\models\RealESRGAN\RealESRGAN_x4plus.pth', 
        ckpt_path = r"E:\Research\symlink\model\PonyV6XL_4base.safetensors", 
        device='cuda'):
    """Initialize RRDBNet and VAE models."""
    # Setup RRDBNet model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                    num_block=23, num_grow_ch=32, scale=4)
    model.to(device)
    
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        dni_weight=None,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        gpu_id=0
    )
    
    # Setup VAE model
    vae = AutoencoderKL.from_single_file(ckpt_path)
    vae.eval()
    vae.to(device)
    
    return upsampler, vae

def get_image_files(folder):
    """Recursively get all image files from folder and subfolders."""
    image_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                image_files.append(os.path.join(root, file))
    return image_files

def process_image(img, speed_scale, upsampler, visibility = 0.75):
    """Process single image with upsampling."""
    h_input, w_input = img.shape[0:2]
    img_input = cv2.resize(
        img, 
        (int(w_input * speed_scale), int(h_input * speed_scale)),
        interpolation=cv2.INTER_LANCZOS4
    )
    output, _ = upsampler.enhance(img_input, )

    outscale=1/speed_scale
    output = cv2.resize(
        output, (
            int(w_input * outscale),
            int(h_input * outscale),
        ), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.resize(
        img, (
            int(w_input * outscale),
            int(h_input * outscale),
        ), interpolation=cv2.INTER_LANCZOS4)
    return (output * visibility + img * (1-visibility)).round().astype(np.uint8)

def save_black_image(height, width, output_path):
    """Save black image with specified dimensions."""
    black_image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.imwrite(output_path, black_image)

def generate_latent(image, vae, device):
    """Generate latent representation using VAE."""
    image_norm = image / 255.0  # Normalize image to [0, 1]
    image_norm = image_norm * 2 - 1
    image_tensor = np.transpose(image_norm, (2, 0, 1))[None, ...]  # HWC -> NCHW
    image_tensor = torch.from_numpy(image_tensor).float()
    
    with torch.no_grad():
        latent = vae.encode(image_tensor.to(device)).latent_dist.sample()
        return latent[0]#.cpu().numpy()
    
def save_latents_to_disk(npz_path, latents_tensor, original_size, crop_ltrb, flipped_latents_tensor=None, alpha_mask=None):
    kwargs = {}
    if flipped_latents_tensor is not None:
        kwargs["latents_flipped"] = flipped_latents_tensor.float().cpu().numpy()
    if alpha_mask is not None:
        kwargs["alpha_mask"] = alpha_mask.float().cpu().numpy()
    np.savez(
        npz_path,
        latents=latents_tensor.float().cpu().numpy(),
        original_size=np.array(original_size),
        crop_ltrb=np.array(crop_ltrb),
        **kwargs,
    )

def trim_and_resize_if_required(
    random_crop: bool, image: np.ndarray, reso, resized_size: Tuple[int, int]
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]:
    image_height, image_width = image.shape[0:2]
    original_size = (image_width, image_height)  # size before resize

    if image_width != resized_size[0] or image_height != resized_size[1]:
        # リサイズする
        image = cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA)  # INTER_AREAでやりたいのでcv2でリサイズ

    image_height, image_width = image.shape[0:2]

    if image_width > reso[0]:
        trim_size = image_width - reso[0]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # logger.info(f"w {trim_size} {p}")
        image = image[:, p : p + reso[0]]
    if image_height > reso[1]:
        trim_size = image_height - reso[1]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # logger.info(f"h {trim_size} {p})
        image = image[p : p + reso[1]]

    # random cropの場合のcropされた値をどうcrop left/topに反映するべきか全くアイデアがない
    # I have no idea how to reflect the cropped value in crop left/top in the case of random crop

    crop_ltrb = BucketManager.get_crop_ltrb(reso, original_size)

    assert image.shape[0] == reso[1] and image.shape[1] == reso[0], f"internal error, illegal trimmed size: {image.shape}, {reso}"
    return image, original_size, crop_ltrb

class BucketManager:
    def __init__(self, no_upscale, max_reso, min_size, max_size, reso_steps) -> None:
        if max_size is not None:
            if max_reso is not None:
                assert max_size >= max_reso[0], "the max_size should be larger than the width of max_reso"
                assert max_size >= max_reso[1], "the max_size should be larger than the height of max_reso"
            if min_size is not None:
                assert max_size >= min_size, "the max_size should be larger than the min_size"

        self.no_upscale = no_upscale
        if max_reso is None:
            self.max_reso = None
            self.max_area = None
        else:
            self.max_reso = max_reso
            self.max_area = max_reso[0] * max_reso[1]
        self.min_size = min_size
        self.max_size = max_size
        self.reso_steps = reso_steps

        self.resos = []
        self.reso_to_id = {}
        self.buckets = []  # 前処理時は (image_key, image, original size, crop left/top)、学習時は image_key

    def add_image(self, reso, image_or_info):
        bucket_id = self.reso_to_id[reso]
        self.buckets[bucket_id].append(image_or_info)

    def shuffle(self):
        for bucket in self.buckets:
            random.shuffle(bucket)

    def sort(self):
        # 解像度順にソートする（表示時、メタデータ格納時の見栄えをよくするためだけ）。bucketsも入れ替えてreso_to_idも振り直す
        sorted_resos = self.resos.copy()
        sorted_resos.sort()

        sorted_buckets = []
        sorted_reso_to_id = {}
        for i, reso in enumerate(sorted_resos):
            bucket_id = self.reso_to_id[reso]
            sorted_buckets.append(self.buckets[bucket_id])
            sorted_reso_to_id[reso] = i

        self.resos = sorted_resos
        self.buckets = sorted_buckets
        self.reso_to_id = sorted_reso_to_id

    #def make_buckets(self):
    #    resos = model_util.make_bucket_resolutions(self.max_reso, self.min_size, self.max_size, self.reso_steps)
    #    self.set_predefined_resos(resos)

    def set_predefined_resos(self, resos):
        # 規定サイズから選ぶ場合の解像度、aspect ratioの情報を格納しておく
        self.predefined_resos = resos.copy()
        self.predefined_resos_set = set(resos)
        self.predefined_aspect_ratios = np.array([w / h for w, h in resos])

    def add_if_new_reso(self, reso):
        if reso not in self.reso_to_id:
            bucket_id = len(self.resos)
            self.reso_to_id[reso] = bucket_id
            self.resos.append(reso)
            self.buckets.append([])
            # logger.info(reso, bucket_id, len(self.buckets))

    def round_to_steps(self, x):
        x = int(x + 0.5)
        return x - x % self.reso_steps

    def select_bucket(self, image_width, image_height):
        aspect_ratio = image_width / image_height
        if not self.no_upscale:
            # 拡大および縮小を行う
            # 同じaspect ratioがあるかもしれないので（fine tuningで、no_upscale=Trueで前処理した場合）、解像度が同じものを優先する
            reso = (image_width, image_height)
            if reso in self.predefined_resos_set:
                pass
            else:
                ar_errors = self.predefined_aspect_ratios - aspect_ratio
                predefined_bucket_id = np.abs(ar_errors).argmin()  # 当該解像度以外でaspect ratio errorが最も少ないもの
                reso = self.predefined_resos[predefined_bucket_id]

            ar_reso = reso[0] / reso[1]
            if aspect_ratio > ar_reso:  # 横が長い→縦を合わせる
                scale = reso[1] / image_height
            else:
                scale = reso[0] / image_width

            resized_size = (int(image_width * scale + 0.5), int(image_height * scale + 0.5))
            # logger.info(f"use predef, {image_width}, {image_height}, {reso}, {resized_size}")
        else:
            # 縮小のみを行う
            if image_width * image_height > self.max_area:
                # 画像が大きすぎるのでアスペクト比を保ったまま縮小することを前提にbucketを決める
                resized_width = math.sqrt(self.max_area * aspect_ratio)
                resized_height = self.max_area / resized_width
                assert abs(resized_width / resized_height - aspect_ratio) < 1e-2, "aspect is illegal"

                # リサイズ後の短辺または長辺をreso_steps単位にする：aspect ratioの差が少ないほうを選ぶ
                # 元のbucketingと同じロジック
                b_width_rounded = self.round_to_steps(resized_width)
                b_height_in_wr = self.round_to_steps(b_width_rounded / aspect_ratio)
                ar_width_rounded = b_width_rounded / b_height_in_wr

                b_height_rounded = self.round_to_steps(resized_height)
                b_width_in_hr = self.round_to_steps(b_height_rounded * aspect_ratio)
                ar_height_rounded = b_width_in_hr / b_height_rounded

                # logger.info(b_width_rounded, b_height_in_wr, ar_width_rounded)
                # logger.info(b_width_in_hr, b_height_rounded, ar_height_rounded)

                if abs(ar_width_rounded - aspect_ratio) < abs(ar_height_rounded - aspect_ratio):
                    resized_size = (b_width_rounded, int(b_width_rounded / aspect_ratio + 0.5))
                else:
                    resized_size = (int(b_height_rounded * aspect_ratio + 0.5), b_height_rounded)
                # logger.info(resized_size)
            else:
                resized_size = (image_width, image_height)  # リサイズは不要

            # 画像のサイズ未満をbucketのサイズとする（paddingせずにcroppingする）
            bucket_width = resized_size[0] - resized_size[0] % self.reso_steps
            bucket_height = resized_size[1] - resized_size[1] % self.reso_steps
            # logger.info(f"use arbitrary {image_width}, {image_height}, {resized_size}, {bucket_width}, {bucket_height}")

            reso = (bucket_width, bucket_height)

        self.add_if_new_reso(reso)

        ar_error = (reso[0] / reso[1]) - aspect_ratio
        return reso, resized_size, ar_error

    @staticmethod
    def get_crop_ltrb(bucket_reso: Tuple[int, int], image_size: Tuple[int, int]):
        # Stability AIの前処理に合わせてcrop left/topを計算する。crop rightはflipのaugmentationのために求める
        # Calculate crop left/top according to the preprocessing of Stability AI. Crop right is calculated for flip augmentation.

        bucket_ar = bucket_reso[0] / bucket_reso[1]
        image_ar = image_size[0] / image_size[1]
        if bucket_ar > image_ar:
            # bucketのほうが横長→縦を合わせる
            resized_width = bucket_reso[1] * image_ar
            resized_height = bucket_reso[1]
        else:
            resized_width = bucket_reso[0]
            resized_height = bucket_reso[0] / image_ar
        crop_left = (bucket_reso[0] - resized_width) // 2
        crop_top = (bucket_reso[1] - resized_height) // 2
        crop_right = crop_left + resized_width
        crop_bottom = crop_top + resized_height
        return crop_left, crop_top, crop_right, crop_bottom

def main():
    # Configuration
    input_folder = 'data/test_in'
    output_folder = 'data/test_out'
    latent_folder = output_folder
    speed_scale = 0.8        # use 1 to compute in full res (slowest)
    is_save_black = False
    device = torch.device('cuda')
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Setup models
    upsampler, vae = setup_models(device=device)
    
    # Get image files
    image_files = get_image_files(input_folder)
    
    # Process images
    start_time = time.time()
    
    for image_path in progress_bar(image_files):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output = process_image(img_rgb, speed_scale, upsampler)
        
        # Generate output paths
        relative_path = os.path.relpath(image_path, input_folder)
        output_path = os.path.join(output_folder, relative_path)
        latent_path = os.path.join(latent_folder, 
                                  f'{os.path.splitext(relative_path)[0]}.npz')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if is_save_black :
            save_black_image(output.shape[0], output.shape[1], output_path)
        else: 
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, output)
        
        h,w,_ = img.shape
        resized_size = (w,h)
        image, original_size, crop_ltrb = trim_and_resize_if_required(False, output, resized_size, resized_size)
        latent = generate_latent(image, vae, device)

        save_latents_to_disk(latent_path,
                             latent,
                             original_size,
                             crop_ltrb,
                             None, None,
                             )
    
    print(f"Total running time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()