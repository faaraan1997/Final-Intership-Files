from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)


model_path = "liuhaotian/llava-v1.5-7b"
prompt = "What method is suggested for addressing noisy in images?"
image_file = "/scratch/faaraan/LLaVAData/images/week_03/week_03_page_013.png"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
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


import sys
from io import StringIO

original_stdout = sys.stdout
sys.stdout = StringIO()

# Call the function that prints directly to stdout
eval_model(args)

# Get the printed output
printed_output = sys.stdout.getvalue()

# Restore original stdout
sys.stdout = original_stdout

# Print the captured output
print("I am here")
print(printed_output)