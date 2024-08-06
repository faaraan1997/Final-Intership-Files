import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import sys
from io import StringIO
import os

# Load your DataFrame (assuming it's a CSV file, adjust as necessary)
Q_A = pd.read_json("/scratch/faaraan/SparrowVQA_Model/MachineLearning/result_9_16.json").head(200)
image_folder = '/scratch/faaraan/SparrowVQA_Model/MachineLearning/images'

Q_A['Image_Path'] = [os.path.join(image_folder, f"week_{week:02d}_page_{page:03d}.png") for week, page in zip(Q_A['week'], Q_A['page'])]


# Correct paths
model_path = "/scratch/faaraan/LLaVA/checkpoints/llava-v1.5-13b-task-lora-ML-Epochs"
model_base = "liuhaotian/llava-v1.5-13b"

try:
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    print("Tokenizer loaded successfully")
except Exception as e:
    print(f"Failed to load tokenizer: {e}")

try:
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_base,
        device_map="auto",  # Automatically map to available devices
        load_in_8bit=False, # Whether to load the model in 8-bit precision
        load_in_4bit=False, # Whether to load the model in 4-bit precision
        use_flash_attn=False, # Whether to use Flash Attention
        trust_remote_code=True, # Required if using custom code from the model hub
    )
    print("Base model loaded successfully")
except Exception as e:
    print(f"Failed to load base model: {e}")

try:
    # Apply the LoRA weights
    model = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=get_model_name_from_path(model_path),
        device="cuda",  # Specify the device
    )
    print("LoRA model loaded successfully")
except Exception as e:
    print(f"Failed to load LoRA model: {e}")

# Function to evaluate model with given prompt and image
def evaluate_model(prompt, image_file):
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": model_base,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    # Capture the printed output of the eval_model function
    original_stdout = sys.stdout
    sys.stdout = StringIO()

    # Call the function that prints directly to stdout
    try:
        eval_model(args)
        printed_output = sys.stdout.getvalue()
        print("Evaluation completed successfully")
    except Exception as e:
        printed_output = sys.stdout.getvalue()
        print(f"Failed during evaluation: {e}")

    # Restore original stdout
    sys.stdout = original_stdout

    # Return the captured output
    return printed_output

# Initialize a list to store results
results = []
import re
# Regular expression pattern to remove unwanted log messages
log_pattern = re.compile(r'Loading LLaVA from base model...\nLoading additional LLaVA weights...\nLoading LoRA weights...\nMerging LoRA weights...\nModel is loaded...\n')


# Loop through the DataFrame and process each entry
for index, entry in Q_A.iterrows():
    # Construct image path based on Image_Path in the DataFrame
    image_path = entry['Image_Path']

    # Get question and original response
    week = entry['Image_Path']
    question = entry['instruction']
    original_response = entry['response']

    # Evaluate model with the current question and image
    predicted_response = evaluate_model(question, image_path)
    # Remove log messages from the output
    predicted_response = log_pattern.sub('', predicted_response)
    print(predicted_response)
    # Append the result to the list
    results.append({
        "week": week,
        "question": question,
        "original_response": original_response,
        "predicted_response": predicted_response
    })

# Save results to a JSON file
output_file = '/scratch/faaraan/LLaVAData/results_from_my_lora_ML_13B.json'  # Adjust the path to your desired output file location
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file}")
