from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from PIL import Image
import requests
import io

# Initialize model and processor
model = AutoModelForCausalLM.from_pretrained("MiaoshouAI/Florence-2-base-PromptGen-v1.5", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("MiaoshouAI/Florence-2-base-PromptGen-v1.5", trust_remote_code=True)

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model=model.to(device)

prompt = "<MIXED_CAPTION>"

#url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
#image = Image.open(requests.get(url, stream=True).raw)

image_path = r"00.png"
image = Image.open(image_path)

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    num_beams=3,
    do_sample=False
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

print(parsed_answer)