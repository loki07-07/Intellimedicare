# app.py
import os
from uuid import uuid4
from pathlib import Path

from flask import Flask, render_template, request, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ------------------- CONFIG -------------------
MODEL_PATH = "skincancer_model.pt"              # your saved model path
IMG_SIZE = 112                                   # must match training
CLASS_NAMES = ["Benign", "Malignant"]            # binary: index 0/1
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}

# Allow forcing CPU if needed: set env var FORCE_CPU=1
FORCE_CPU = os.getenv("FORCE_CPU", "0") == "1"
device = torch.device("cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda:0")

predict_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

app = Flask(__name__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR


# ------------------- MODEL LOAD -------------------
def load_full_model_safely(path: str, device: torch.device):
    """
    Loads a model saved via torch.save(model, path) robustly:
    - always map to CPU first to avoid device mismatches,
    - unwrap DataParallel/DistributedDataParallel if present,
    - then move to target device.
    """
    # 1) Load to CPU first
    mdl = torch.load(path, map_location="cpu")

    # 2) Unwrap wrappers if present
    if isinstance(mdl, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        mdl = mdl.module

    # 3) Move to target device
    mdl.to(device)
    mdl.eval()
    return mdl

try:
    model = load_full_model_safely(MODEL_PATH, device)
    print(f"✅ Loaded model from {MODEL_PATH} on device: {device}")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model from {MODEL_PATH}: {e}")


# ------------------- HELPERS -------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def predict_image(pil_img: Image.Image):
    """
    Returns (pred_idx, pred_label, confidence).
    Supports either a single-logit head (sigmoid) or a 2-logit head (softmax).
    Ensures input is placed on the SAME device as the model.
    """
    with torch.no_grad():
        x = predict_transform(pil_img).unsqueeze(0)

        # Make sure input and model share device
        x = x.to(device, non_blocking=True)

        out = model(x)

        # If your head outputs a single logit (shape [B,1]):
        if out.ndim == 2 and out.shape[-1] == 1:
            prob_malignant = torch.sigmoid(out).item()
            pred_idx = int(prob_malignant > 0.5)  # 0=Benign, 1=Malignant
            pred_label = CLASS_NAMES[pred_idx]
            confidence = prob_malignant if pred_idx == 1 else (1 - prob_malignant)
            return pred_idx, pred_label, float(confidence)

        # Else assume two logits (shape [B,2]):
        prob = F.softmax(out, dim=-1).squeeze(0)  # [2]
        conf, pred_idx_t = torch.max(prob, dim=-1)
        pred_idx = int(pred_idx_t.item())
        pred_label = CLASS_NAMES[pred_idx]
        return pred_idx, pred_label, float(conf.item())


# ------------------- ROUTES -------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", result=None, error="No file uploaded.")
    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", result=None, error="No file selected.")
    if not allowed_file(file.filename):
        return render_template("index.html", result=None, error="Unsupported file type.")

    # Save upload
    filename = secure_filename(file.filename)
    filename = f"{uuid4().hex}_{filename}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        img = Image.open(save_path).convert("RGB")
        pred_idx, pred_label, conf = predict_image(img)

        result = {
            "filename": filename,
            "image_url": url_for("serve_upload", filename=filename),
            "pred_idx": pred_idx,
            "pred_label": pred_label,
            "confidence": round(conf, 4),
            "device": str(device),
        }
        return render_template("index.html", result=result, error=None)
    except Exception as e:
        return render_template("index.html", result=None, error=f"Prediction failed: {e}")


@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)



# -------- Optional: JSON API endpoint --------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid or unsupported file."}), 400

    filename = secure_filename(file.filename)
    filename = f"{uuid4().hex}_{filename}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    try:
        img = Image.open(path).convert("RGB")
        pred_idx, pred_label, conf = predict_image(img)
        return jsonify({
            "pred_idx": pred_idx,
            "pred_label": pred_label,
            "confidence": conf,
            "device": str(device),
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
