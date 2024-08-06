import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import gradio as gr

#Create model
model = AutoModelForCausalLM.from_pretrained(
    "rrymn/SparrowVQE",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("rrymn/SparrowVQE", trust_remote_code=True)



# # Ensure GPU usage if available
device = "cuda" if torch.cuda.is_available() else "cpu"

def predict_answer(image, question, max_tokens=10):
    #Set inputs
    text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{question}? ASSISTANT:"
    image = image.convert("RGB")

    input_ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
    image_tensor = model.image_preprocess(image)

    #Generate the answer
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        images=image_tensor,
        use_cache=True)[0]

    return tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

def gradio_predict(image, question, max_tokens):
    answer = predict_answer(image, question, max_tokens)
    return answer

# Define the Gradio interface
iface = gr.Interface(
    fn=gradio_predict,
    inputs=[gr.Image(type="pil", label="Upload or Drag an Image"),
            gr.Textbox(label="Question", placeholder="e.g. Can you explain the slide?", scale=4),
            gr.Slider(2, 500, value=100, label="Token Count", info="Choose between 2 and 500")],
    outputs=gr.TextArea(label="Answer"),
    # examples=examples,
    title="SparrowVQE",
    description="An interactive chat model that can answer questions about images in an Academic context. \n We can input images, and the system will analyze them to provide information about their contents. I've utilized this capability by feeding slides from PowerPoint presentations used in classes and the lecture content passed as text. Consequently, the model now mimics the behavior and responses of my professors. So, if I present any PowerPoint slide, it explains it just like my professor would, further it can be personalized.",
)

# Launch the app
iface.queue().launch(debug=True)