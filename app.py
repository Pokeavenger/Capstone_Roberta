import os
import torch
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "SharvNey/capstone_project"

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=os.getenv("HF_TOKEN"))
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, use_auth_token=os.getenv("HF_TOKEN"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prediction function
def classify_text(text):
    if not text.strip():
        return {"🧑 Human-Written": 0.0, "🤖 AI-Generated": 0.0}

    enc = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        probs = torch.nn.functional.softmax(out.logits, dim=-1).cpu().numpy()[0]

    return {"🧑 Human-Written": float(probs[0]), "🤖 AI-Generated": float(probs[1])}

# Gradio app
demo = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(lines=8, placeholder="Paste text here..."),
    outputs=gr.Label(num_top_classes=2),
    title="🤖 AI vs Human Text Classifier",
    description="Fine-tuned RoBERTa model that detects whether text is Human-written 🧑 or AI-generated 🤖"
)

if __name__ == "__main__":
    # Detect if running on Hugging Face Spaces
    in_spaces = os.environ.get("SYSTEM") == "spaces"

    if in_spaces:
        demo.launch(server_name="0.0.0.0", server_port=7860)
    else:
        demo.launch(share=True)
