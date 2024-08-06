from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import sys
from io import StringIO

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)


import json

# Function to get model name from path (placeholder)
# def get_model_name_from_path(model_path):
#     return model_path.split('/')[-1]

# # Placeholder for eval_model function
# def eval_model(args):
#     # Implement the actual model evaluation logic here
#     pass

# Placeholder for caption_image function
def caption_image(image_path, question):
    # Implement the actual image captioning logic here
    return image_path, "Generated answer based on image and question"

# # Load JSON data
# with open('Final_Week_Part_2.json', 'r', encoding="UTF-8") as f:
#     data = json.load(f)

# List to store output for each entry
output_data = []
image_folder = '/scratch/faaraan/LLaVAData/images/'
import pandas as pd
import os

Q_A = pd.read_json("/scratch/faaraan/LLaVAData/Q_A.json")

Q_A['Image_Path'] = [os.path.join(image_folder, f"week_{week:02d}/week_{week:02d}_page_{page:03d}.png") for week, page in zip(Q_A['week'], Q_A['page'])]


# Iterate through each entry in the JSON data
for index, entry in Q_A.iterrows():
    # print("Entry")
    print(entry['Image_Path'])
    # Construct image path based on week and page number
    # if entry['page'] < 10:
    #     image_path = f'/scratch/faaraan/LLaVAData/images/week_0{entry["week"]}/week_0{entry["week"]}_page_00{entry["page"]}.png'
    # elif entry['page'] > 100:
    #     image_path = f'/scratch/faaraan/LLaVAData/images/week_0{entry["week"]}/week_0{entry["week"]}_page_{entry["page"]}.png'
    # else:
    #     image_path = f'/scratch/faaraan/LLaVAData/images/week_0{entry["week"]}/week_0{entry["week"]}_page_{entry["page"]}.png'

    # Get question, answer, and original response
    question = entry['instruction']
    original_response = entry['response']

    # Prepare arguments for eval_model
    args = type('Args', (), {
        "model_path": "liuhaotian/llava-v1.5-7b",
        "model_base": None,
        "model_name": get_model_name_from_path("liuhaotian/llava-v1.5-7b"),
        "query": question,
        "conv_mode": None,
        "image_file": entry['Image_Path'],
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    original_stdout = sys.stdout
    sys.stdout = StringIO()
    eval_model(args)
    # Get the printed output
    answer = sys.stdout.getvalue()

    # Restore original stdout
    sys.stdout = original_stdout

    # Print the required information
    print("Question:", question)
    print("Image Path:", entry['Image_Path'])
    print("Predicted Answer:", answer)
    print("Original Response:", original_response)
    print("\n")

    output_entry = {
        "question": question,
        "image_path": entry['Image_Path'],
        "predicted_answer": answer,
        "original_response": original_response
    }

    output_data.append(output_entry)

# Save the output to a JSON file
output_file = 'output.json'
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=4)

print("Output saved to:", output_file)
