import os
import json
import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from transformers import TextStreamer

# Correct paths
questions_file = "/scratch/faaraan/SparrowVQA_Model/MachineLearning/result_9_16.json"
image_folder = '/scratch/faaraan/SparrowVQA_Model/MachineLearning/images'
output_file = '/scratch/faaraan/LLaVAData/results_from_my_lora_ML.json'
model_path = "/scratch/faaraan/LLaVA/checkpoints/llava-v1.5-7b-task-lora-ml"
model_base = "liuhaotian/llava-v1.5-7b"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
kwargs = {"device_map": "auto"}
kwargs['load_in_4bit'] = True
kwargs['quantization_config'] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

# Load vision tower
vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device='cuda')
image_processor = vision_tower.image_processor

# Function to caption image with the model
def caption_image(image_file, prompt):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        if not os.path.exists(image_file):
            print(f"Image file does not exist: {image_file}")
            return None, "Image file not found"
        image = Image.open(image_file).convert('RGB')
    # image = Image.open(image_file).convert('RGB')

    disable_torch_init()
    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    inp = f"{roles[0]}: {prompt}"
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    raw_prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2,
                                    max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs
    output = outputs.rsplit('</s>', 1)[0]
    
    return image, output

# Load questions from JSON file
with open(questions_file, encoding='utf-8') as f:
    questions_data = json.load(f)

import pandas as pd
results = []
Q_A = pd.read_json("/scratch/faaraan/SparrowVQA_Model/MachineLearning/result_9_16.json").head(4)
image_folder = '/scratch/faaraan/SparrowVQA_Model/MachineLearning/images'

Q_A['Image_Path'] = [os.path.join(image_folder, f"week_{week:02d}_page_{page:03d}.png") for week, page in zip(Q_A['week'], Q_A['page'])]


# Loop through each entry in the JSON file
for index, data in Q_A.iterrows():
    image_file = f"week_{data['week']:02d}_page_{data['page']:03d}.png"
    image_path = os.path.join(image_folder, image_file)
    question = data['instruction']
    ground_truth = data['response']
    image, output = caption_image(data['Image_Path'], question)

    if image is None:
        continue
    print(output)
    result = {
        'image': image_file,
        'question': question,
        'output': output,
        'ground_truth': ground_truth
    }
    results.append(result)

# Save results to JSON file
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file}")
