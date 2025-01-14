# caption with moondream2

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import os
from fastprogress import master_bar, progress_bar

model_id = "vikhyatk/moondream2"
revision = "2b705eea63f9bff6dae9b52c2daeb26bc10e4aeb" #"2b705eea63f9bff6dae9b52c2daeb26bc10e4aeb" #"2024-03-05"


# Check if CUDA (GPU support) is available and then set the device to GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, 
                                             revision=revision, attn_implementation="flash_attention_2").to(device)
                                             #revision=revision,).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

img_dir = r"E:\Research\symlink\CivitAI\anime\dandadan\draft\0other"
prompt = "janninew"

img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png") or f.endswith(".jpg")])


count = 0

# Loop through each image in the directory
for img_file in progress_bar(img_files):
    img_path = os.path.join(img_dir, img_file)
    image = Image.open(img_path)
    #print(img_file)
    
    # Resize the image if needed
    if 0:
        image = image.resize((224, 224))  # Adjust the size as per your model requirements
    elif 0:
        width, height = image.size
        # Crop the left half of the image
        image = image.crop((0, 0, width // 2, height))
        image = image.resize((192, 192))
    elif 1:
        width, height = image.size
        image = image.resize((width//4, height//4))

    # Convert the image to RGB if it's not
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Encode the image
    enc_image = model.encode_image(image).to(device)
    
    # Generate the answer
    #answer = model.answer_question(enc_image, "give me a short caption of this pokemon", tokenizer)
    #answer = model.answer_question(enc_image, "Write a short caption", tokenizer, chat_history="Let call it as Pokemon. ")
    #answer = model.answer_question(enc_image, "Write a short caption", tokenizer, chat_history="")
    #answer = model.answer_question(enc_image, "Write a short caption to describe person", tokenizer, chat_history="")
    #answer = model.answer_question(enc_image, "Descript the outfit, attire, and accessories shortly", tokenizer, chat_history="")
    answer = model.answer_question(enc_image, "Descript the image shortly", tokenizer, chat_history="")

    # If the output is a tensor, move it back to CPU for further operations like print
    if isinstance(answer, torch.Tensor):
        answer = answer.cpu().item()
    
    # Save the answer in a text file with the same name as the image
    with open(os.path.splitext(img_path)[0] + ".txt", "w") as f:
        f.write(str(answer))
        #f.write(f"{prompt}, "+str(answer))
    #print(answer)
        
    # Update master progress bar
    #mb.write(f"Processed: {img_file}", table=True)
    #mb.child.comment = f"Processed: {img_file}"
    #mb.update()
    count = count + 1
    #if count > 10 : break