import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from fastprogress import master_bar, progress_bar

def is_image_file(filename):
    """Check if the file is an image based on its extension."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    return os.path.splitext(filename)[1].lower() in image_extensions


def find_image_files_ancestor(folder_path):
    image_files = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if is_image_file(filename):
                full_image_path = os.path.join(root, filename)
                image_files.append(full_image_path)
    return image_files

def find_image_files(folder_path, overwrite_ok = False):
    image_files = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if is_image_file(filename):
                full_image_path = os.path.join(root, filename)

                # Get the corresponding JSON filename
                base_name = os.path.splitext(filename)[0]
                json_filename = base_name + '.json'
                json_path = os.path.join(root, json_filename)
                
                if not os.path.exists(json_path) or overwrite_ok:
                    image_files.append(full_image_path)

    return image_files

def batch_process_images(image_paths, model, processor, device, prompt, batch_size=16, mb=None):
    """
    Process images in batches with a single prompt.
    
    Args:
        image_paths: List of image file paths
        model: The Florence-2 model
        processor: The model processor
        device: Torch device (cuda or cpu)
        prompt: Prompt for generation
        batch_size: Number of images to process in a single batch
    
    Returns:
        List of generated results
    """
    # Initialize results list
    results = []
    
    # Create a master progress bar
    #mb = master_bar(range(0, len(image_paths), batch_size))
    
    for start in progress_bar(range(0, len(image_paths), batch_size), parent=mb):
        # Slice the batch
        batch_paths = image_paths[start:start+batch_size]
        
        # Prepare batch inputs
        inputs = {
            "input_ids": [],
            "pixel_values": []
        }
        
        # Prepare images for the batch
        for image_path in batch_paths:
            # Open and convert image
            image = Image.open(image_path).convert("RGB")
            
            # Prepare inputs
            input_data = processor(
                text=prompt, 
                images=image, 
                return_tensors="pt", 
                do_rescale=False
            )
            
            inputs["input_ids"].append(input_data["input_ids"])
            inputs["pixel_values"].append(input_data["pixel_values"])

        # Concatenate inputs
        inputs["input_ids"] = torch.cat(inputs["input_ids"]).to(device)
        inputs["pixel_values"] = torch.cat(inputs["pixel_values"]).to(device).to(torch.bfloat16)
        
        # Generate responses
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
        
        # Decode results
        skip_special_tokens = prompt in ["<MIXED_CAPTION>"]
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Post-process generations
        batch_results = []
        for text, image_path in zip(generated_texts, batch_paths):
            # Post-process the generation
            parsed_answer = processor.post_process_generation(
                text, 
                task=prompt, 
                # Use a default image size if needed
                image_size=(224, 224)
            )
            
            # Extract the result for the specific prompt
            result = parsed_answer.get(prompt, "")
            batch_results.append((image_path, result))
                
        
        # Extend overall results
        results.extend(batch_results)
        
    
    return results

def process_images_with_prompts(image_files, model, processor, device):

    prompts = {
        "<CAPTION>": "pg_cap",
        "<GENERATE_TAGS>": "pg_tag",
        "<MIXED_CAPTION>": "pg_mixed",
        "<ANALYZE>": "pg_analyze",
        "<MIXED_CAPTION_PLUS>": "pg_mixed_plus",
        "<MORE_DETAILED_CAPTION>": "pg_md_cap",
    }
    
    mb = master_bar(prompts.items())
    
    for prompt, key in mb:
        print(f"Processing prompt: {prompt}")
        results = batch_process_images(image_files, model, processor, device, prompt, mb=mb)
        
        for image_path, result in results:
            json_path = os.path.splitext(image_path)[0] + ".json"
            image_results = {}
            
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    image_results = json.load(f)
            
            image_results[key] = result

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(image_results, f, ensure_ascii=False, indent=4)

def process_and_save_images(image_files, model, processor, batch_size=4, prompts = None):
    if prompts is None:
        prompts = {
            "<CAPTION>": "pg_cap",
            "<GENERATE_TAGS>": "pg_tag",
            #"<MIXED_CAPTION>": "pg_mixed",
            #"<ANALYZE>": "pg_analyze",
            #"<MIXED_CAPTION_PLUS>": "pg_mixed_plus",
            #"<MORE_DETAILED_CAPTION>": "pg_md_cap",
        }
    mb = master_bar(prompts.items())

    device = model.device
    for prompt, key in mb:
        print(f"Processing prompt: {prompt}")
        for start in progress_bar(range(0, len(image_files), batch_size), parent=mb):
            batch_paths = image_files[start:start + batch_size]
            inputs = {"input_ids": [], "pixel_values": []}

            for image_path in batch_paths:
                image = Image.open(image_path).convert("RGB")
                input_data = processor(
                    text=prompt, images=image, return_tensors="pt", do_rescale=False
                )
                inputs["input_ids"].append(input_data["input_ids"])
                inputs["pixel_values"].append(input_data["pixel_values"])

            inputs = {
                "input_ids": torch.cat(inputs["input_ids"]).to(device),
                "pixel_values": torch.cat(inputs["pixel_values"]).to(device).to(torch.bfloat16),
            }

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                )
            skip_special_tokens = prompt in ["<MIXED_CAPTION>"]
            generated_texts = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            for text, image_path in zip(generated_texts, batch_paths):
                parsed_answer = processor.post_process_generation(
                    text, task=prompt, image_size=(224, 224)
                )
                result = parsed_answer.get(prompt, "")
                json_path = os.path.splitext(image_path)[0] + ".json"

                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        image_results = json.load(f)
                else:
                    image_results = {}

                image_results[key] = result

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(image_results, f, ensure_ascii=False, indent=4)

def load_model(model_name):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device)
    return model, processor
    
if __name__ == "__main__":
    # Configuration
    folder_path = r"E:\Research\symlink\CivitAI\disney\realcartoon\draft"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    image_files = find_image_files(folder_path, overwrite_ok=True)
    print(f"Found {len(image_files)} image files to process")

    model, processor = load_model("MiaoshouAI/Florence-2-large-PromptGen-v2.0")
    process_and_save_images(image_files, model, processor, prompts={"<CAPTION>": "pg_cap"})

    #model, processor = load_model("MiaoshouAI/Florence-2-large-PromptGen-v1.5")
    #model, processor = load_model("MiaoshouAI/Florence-2-base-PromptGen-v1.5")
    #process_and_save_images(image_files, model, processor, prompts={"<GENERATE_TAGS>": "pg_tag"})

    print("Processing complete!")
