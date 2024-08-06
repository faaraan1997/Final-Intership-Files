import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import pandas as pd
import os
from rouge import Rouge
from tqdm import tqdm

torch.set_default_device("cuda")

# Load your DataFrame (adjust the path as necessary)
# Q_A = pd.read_json("/scratch/faaraan/SparrowVQA_Model/test_sparrow_data.json")
Q_A = pd.read_json("/scratch/faaraan/LLaVAData/data_week_9_to_16.json")

image_folder = '/scratch/faaraan/LLaVAData/images/'

Q_A['Image_Path'] = [os.path.join(image_folder, f"week_{week:02d}/week_{week:02d}_page_{page:03d}.png") for week, page in zip(Q_A['week'], Q_A['page'])]

# Load fine-tuned model
model_path = "/scratch/faaraan/SparrowVQA_Model/Sparrowvqe/checkpoints/SparrowVQE-3b-stage6"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("rrymn/SparrowVQE", trust_remote_code=True)

def predict(image_path, question):
    # Set inputs
    text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{question}? ASSISTANT:"
    image = Image.open(image_path)
    
    input_ids = tokenizer(text, return_tensors='pt').input_ids.to('cuda')
    image_tensor = model.image_preprocess(image)
    
    # Generate the answer
    output_ids = model.generate(
        input_ids,
        max_new_tokens=150,
        images=image_tensor,
        use_cache=True)[0]
    
    return tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

def calculate_rouge(predicted_text, ground_truth_text):
    rouge = Rouge()
    scores = rouge.get_scores(predicted_text, ground_truth_text)
    rouge_1_score = scores[0]["rouge-1"]["f"]
    rouge_2_score = scores[0]["rouge-2"]["f"]
    rouge_l_score = scores[0]["rouge-l"]["f"]
    return rouge_1_score, rouge_2_score, rouge_l_score

# Lists to store individual ROUGE scores
rouge_scores = []

# Iterate through the DataFrame to get images and questions and predict answers
for idx, row in tqdm(Q_A.iterrows(), total=Q_A.shape[0], desc="Processing"):
    image_path = row['Image_Path']
    predicted_answer = predict(image_path, row['instruction'])
    reference_answer = row['response']

    # Calculate ROUGE scores
    rouge_1, rouge_2, rouge_l = calculate_rouge(predicted_answer, reference_answer)
    
    # Store the predicted answer and ROUGE scores
    Q_A.at[idx, 'Predicted_Answer'] = predicted_answer
    Q_A.at[idx, 'Image Path'] = image_path

    rouge_scores.append({'ROUGE-1': rouge_1, 'ROUGE-2': rouge_2, 'ROUGE-L': rouge_l})

# Convert the ROUGE scores to a DataFrame
rouge_df = pd.DataFrame(rouge_scores)

# Calculate average ROUGE scores
average_rouge_scores = rouge_df.mean()

print("Average ROUGE-1 score:", average_rouge_scores['ROUGE-1'])
print("Average ROUGE-2 score:", average_rouge_scores['ROUGE-2'])
print("Average ROUGE-L score:", average_rouge_scores['ROUGE-L'])

# Save the DataFrame with predicted answers and ROUGE scores
Q_A_with_rouge = pd.concat([Q_A, rouge_df], axis=1)
Q_A_with_rouge.to_json('/scratch/faaraan/SparrowVQA_Model/Q_A_with_predictions_and_rouge.json', orient='records', lines=True)
