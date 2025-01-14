from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from PIL import Image

# Initialize model and processor
model = AutoModelForCausalLM.from_pretrained("MiaoshouAI/Florence-2-base-PromptGen-v1.5", trust_remote_code=True, torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained("MiaoshouAI/Florence-2-base-PromptGen-v1.5", trust_remote_code=True)

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model=model.to(device)

#prompt = "<MORE_DETAILED_CAPTION>"
#prompt = "<GENERATE_TAGS>"
#prompt = "<MIXED_CAPTION>"
prompt = "<CAPTION>"

image_path = r"car.png"
image_path = r"E:\Research\symlink\CivitAI\anime\dandadan\draft\momoayase\29068527.jpg"
images = [
    r"car.png",
    r"E:\Research\symlink\CivitAI\anime\dandadan\draft\momoayase\29068527.jpg",
    r"E:\Research\symlink\CivitAI\anime\dandadan\draft\momoayase\29068524.jpg",
    r"E:\Research\symlink\CivitAI\anime\dandadan\draft\momoayase\29068895.jpg"
]

image = Image.open(image_path)
image = image.convert("RGB")

#inputs = processor(text=["<CAPTION>","<GENERATE_TAGS>"], images=[image,image], padding=True, return_tensors="pt").to(device)
#inputs = processor(text=["<CAPTION>"], images=[image], padding=True, return_tensors="pt").to(device)
inputs = {
    "input_ids": [],
    "pixel_values" : []
}
#prompts = ["<CAPTION>","<GENERATE_TAGS>"]
prompt = "<CAPTION>"

for image_path in image_path_list:
    image = Image.open(image_path)
    image = image.convert("RGB")

    input_data = processor(text=prompt, images=image, return_tensors="pt", do_rescale=False)
    inputs["input_ids"].append(input_data["input_ids"])
    inputs["pixel_values"].append(input_data["pixel_values"])

inputs["input_ids"] = torch.cat(inputs["input_ids"]).to(device)
inputs["pixel_values"] = torch.cat(inputs["pixel_values"]).to(device).to(torch.bfloat16)

#text = processor._construct_prompts("<CAPTION>")
#inputs = processor.tokenizer(text)
generated_ids = model.generate(
    #input_ids=inputs["input_ids"],
    #pixel_values=inputs["pixel_values"],
    **inputs,
    max_new_tokens=1024,
    num_beams=3,
    do_sample=False
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)#[0]
breakpoint()
parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))

print(generated_text)
chunk_size = 100  # Define how many characters per chunk
for i in range(0, len(parsed_answer[prompt]), chunk_size):
    print(parsed_answer[prompt][i:i+chunk_size])