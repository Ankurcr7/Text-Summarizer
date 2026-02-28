import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify, render_template
import torch
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# =========================
# Load Model from Hugging Face
# =========================
MODEL = "Ankurcr7/Text_summarizer_model"

tokenizer = T5Tokenizer.from_pretrained(MODEL)
model = T5ForConditionalGeneration.from_pretrained(MODEL)

device = torch.device("cpu")
model.to(device)
model.eval()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def generate_summary(text):
    text = text[:1500]  # prevent memory issues

    input_text = "summarize: " + clean_text(text)

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=256,
        truncation=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=120,
            num_beams=1  # reduce memory
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    summary = generate_summary(data["text"])
    return jsonify({"summary": summary})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)