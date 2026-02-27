import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify, render_template
import torch
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

MODEL= "text_summarizer_model_2/text_summarizer_model"

model = T5ForConditionalGeneration.from_pretrained(MODEL)
tokenizer = T5Tokenizer.from_pretrained(MODEL)

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
    input_text = "summarize: " + clean_text(text)

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=250,
            num_beams=6,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True
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
    app.run(debug=True)