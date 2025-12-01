import os
import io
import json
import requests
import numpy as np
import torch
from PIL import Image
from timm import create_model
from flask import Flask, request, jsonify
from transformers import pipeline


# ------------------------------------------------------------
# Load Model From HuggingFace (only once at startup)
# ------------------------------------------------------------

print("🔄 Loading WildArabia model from HuggingFace...")

repo = "Sarahkalfoot/WildArabia"

# Load classes
classes_url = f"https://huggingface.co/{repo}/resolve/main/Classes.json"
classes = json.loads(requests.get(classes_url).text)

# Load preprocessor config
pre_url = f"https://huggingface.co/{repo}/resolve/main/preprocessor_config.json"
pre_cfg = json.loads(requests.get(pre_url).text)

# Download model weights
weights_url = f"https://huggingface.co/{repo}/resolve/main/model_state_dict.pth"
weights_path = "/tmp/model_state_dict.pth"

with open(weights_path, "wb") as f:
    f.write(requests.get(weights_url).content)

# Build ConvNeXt Base model
model = create_model("convnext_base", pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()

print("✅ Model loaded successfully!")


# ------------------------------------------------------------
# Preprocessing (read from preprocessor_config.json)
# ------------------------------------------------------------

def preprocess_image_bytes(img_bytes, cfg):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    if cfg["do_resize"]:
        size = cfg["size"]["shortest_edge"]
        img = img.resize((size, size))

    if cfg.get("do_center_crop", False):
        w, h = img.size
        cw, ch = cfg["crop_size"]["width"], cfg["crop_size"]["height"]
        left = (w - cw) // 2
        top = (h - ch) // 2
        img = img.crop((left, top, left + cw, top + ch))

    arr = np.array(img).astype("float32") / 255.0
    arr = torch.tensor(arr).permute(2, 0, 1)

    if cfg["do_normalize"]:
        mean = torch.tensor(cfg["image_mean"]).unsqueeze(1).unsqueeze(2)
        std = torch.tensor(cfg["image_std"]).unsqueeze(1).unsqueeze(2)
        arr = (arr - mean) / std

    return arr.unsqueeze(0)


# ------------------------------------------------------------
# Initialize Flask + RAG
# ------------------------------------------------------------

app = Flask(__name__)
rag_pipe = pipeline("question-answering", model="deepset/roberta-base-squad2")


@app.route("/")
def home():
    return jsonify({"message": "WildArabia API is running!"})


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    question = request.form.get("question", "").strip()

    if not file:
        return jsonify({"error": "No image provided"}), 400

    try:
        img_bytes = file.read()

        # Step 1 — classify image with your ConvNeXt model
        inputs = preprocess_image_bytes(img_bytes, pre_cfg)

        with torch.no_grad():
            logits = model(inputs)
            pred_id = logits.argmax().item()

        label = classes[str(pred_id)]
        base_context = f"The image contains a {label}."

        # Step 2 — if user asked a question, use RAG
        if question:
            rag_answer = rag_pipe(question=question, context=base_context)
            answer = rag_answer.get("answer", "Could not determine.")
        else:
            answer = base_context

        return jsonify({
            "scientific_name": label,
            "description": answer,
            "context": base_context,
            "question": question or None
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
