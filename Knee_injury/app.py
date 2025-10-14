import os
import io
import csv
from uuid import uuid4
from pathlib import Path

from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, send_file, redirect, url_for
)

import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
from torchvision import transforms

# your project modules
from model import MRNet
from utils import preprocess_data

# ========= CONFIG =========
CNN_MODELS_LIST = "cnn_models_paths.txt"   # 9 lines: 3 per task (ax, cor, sag)
LR_MODELS_LIST  = "lr_models_paths.txt"    # 3 lines: abn, acl, men
FORCE_CPU = os.getenv("FORCE_CPU", "0") == "1"
DEVICE = torch.device("cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_NPY = {".npy"}
ALLOWED_CSV = {".csv"}

# If you normalized during training, add Normalize below.
SLICE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
# =========================


# ----- helpers -----
def ensure_3ch(t: torch.Tensor) -> torch.Tensor:
    """ Ensure [N,3,H,W]. Accepts [N,H,W] or [N,1,H,W] or [N,3,H,W]. """
    if t.ndim == 3:  # [N,H,W]
        t = t.unsqueeze(1)
    if t.shape[1] == 1:
        t = t.repeat(1, 3, 1, 1)
    return t


def allowed_ext(filename: str, allowed: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {x.strip(".") for x in allowed}


def save_upload(file_storage) -> str:
    fname = f"{uuid4().hex}_{file_storage.filename}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    file_storage.save(path)
    return path


# ----- load models once -----
def load_mrnet_models():
    with open(CNN_MODELS_LIST, "r") as f:
        cnn_paths = [line.strip() for line in f if line.strip()]
    if len(cnn_paths) != 9:
        raise RuntimeError(f"Expected 9 paths in {CNN_MODELS_LIST}, found {len(cnn_paths)}")

    with open(LR_MODELS_LIST, "r") as f:
        lr_paths = [line.strip() for line in f if line.strip()]
    if len(lr_paths) != 3:
        raise RuntimeError(f"Expected 3 paths in {LR_MODELS_LIST}, found {len(lr_paths)}")

    abnormal, acl, meniscus = [], [], []
    for i, p in enumerate(cnn_paths):
        m = MRNet().to("cpu")
        ckpt = torch.load(p, map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        state = {k.replace("module.", ""): v for k, v in state.items()}
        m.load_state_dict(state, strict=False)
        m.eval().to(DEVICE)
        if i < 3:
            abnormal.append(m)
        elif i < 6:
            acl.append(m)
        else:
            meniscus.append(m)

    lrs = [joblib.load(p) for p in lr_paths]
    # grouped by task
    return [abnormal, acl, meniscus], lrs


MRNETS, LRS = load_mrnet_models()

def preprocess_data(npy_path, transform):
    data = np.load(npy_path)
    if data.ndim == 4 and data.shape[1] == 1:
        data = data.squeeze(1)
    if data.ndim != 3:
        raise ValueError(f"Expected [N, H, W] or [N, 1, H, W], got shape {data.shape}")
    slices = [transform(slice) for slice in data]
    return torch.stack(slices)  # [N, 3, H, W]

# ----- core predict (your logic, hardened) -----
def predict_case_from_arrays(axial_t, coronal_t, sagittal_t, cnn_models, lr_models, device="cpu"):
    for task in cnn_models:
        for m in task:
            m.to(device).eval()

    plane_inputs = [axial_t, coronal_t, sagittal_t]
    plane_logits = []

    with torch.no_grad():
        for plane_input, plane_models in zip(plane_inputs, list(zip(*cnn_models))):
            t = plane_input  # [1, N, 3, H, W]
            if t.ndim == 5:
                t = t.squeeze(0)  # [N, 3, H, W]
            print(f"Input shape before ensure_3ch: {t.shape}")
            t = ensure_3ch(t).to(device).float()  # [N, 3, H, W]
            print(f"Input shape to model: {t.shape}")
            outs = []
            for m in plane_models:
                o = m(t)  # [N, 1] or [N]
                o = o.reshape(o.shape[0], -1).mean(dim=0)
                outs.append(float(o.detach().cpu().numpy().squeeze()))
            plane_logits.append(outs)

    logits_per_task = list(zip(*plane_logits))
    probs = []
    for task_logits, lr in zip(logits_per_task, lr_models):
        X = np.array(task_logits, dtype=np.float32).reshape(1, -1)
        p = lr.predict_proba(X)[0, 1]
        probs.append(float(p))
    return probs


# ===================== ROUTES =====================
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None, error=None, device=str(DEVICE))


# ---- single-case: upload 3 npy volumes ----
@app.route("/predict", methods=["POST"])
def predict_single():
    need = ("axial", "coronal", "sagittal")
    if any(k not in request.files for k in need):
        return render_template("index.html", result=None, error="Upload axial, coronal, sagittal .npy files.", device=str(DEVICE))

    axial_f    = request.files["axial"]
    coronal_f  = request.files["coronal"]
    sagittal_f = request.files["sagittal"]

    if any(f.filename == "" for f in (axial_f, coronal_f, sagittal_f)):
        return render_template("index.html", result=None, error="One or more files not selected.", device=str(DEVICE))
    if not all(allowed_ext(f.filename, ALLOWED_NPY) for f in (axial_f, coronal_f, sagittal_f)):
        return render_template("index.html", result=None, error="Only .npy files are supported.", device=str(DEVICE))

    ax_path = save_upload(axial_f)
    co_path = save_upload(coronal_f)
    sa_path = save_upload(sagittal_f)

    try:
        axial    = preprocess_data(ax_path, SLICE_TRANSFORM)     # [N,C?,H,W]
        coronal  = preprocess_data(co_path, SLICE_TRANSFORM)
        sagittal = preprocess_data(sa_path, SLICE_TRANSFORM)

        axial    = ensure_3ch(axial).unsqueeze(0).to(DEVICE)     # [1,N,3,H,W]
        coronal  = ensure_3ch(coronal).unsqueeze(0).to(DEVICE)
        sagittal = ensure_3ch(sagittal).unsqueeze(0).to(DEVICE)

        probs = predict_case_from_arrays(axial, coronal, sagittal, MRNETS, LRS, device=DEVICE)
        result = {"P_abnormal": round(probs[0], 4),
                  "P_acl": round(probs[1], 4),
                  "P_meniscus": round(probs[2], 4)}
        return render_template("index.html", result=result, error=None, device=str(DEVICE))
    except Exception as e:
        return render_template("index.html", result=None, error=f"Prediction failed: {e}", device=str(DEVICE))


# ---- batch mode: upload valid-paths.csv -> return predictions.csv ----
@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "GET":
        return render_template("batch.html", error=None, device=str(DEVICE))

    # POST
    if "csv" not in request.files:
        return render_template("batch.html", error="Upload a valid-paths.csv", device=str(DEVICE))
    csv_f = request.files["csv"]
    if csv_f.filename == "" or not allowed_ext(csv_f.filename, ALLOWED_CSV):
        return render_template("batch.html", error="Please upload a .csv file.", device=str(DEVICE))

    csv_path = save_upload(csv_f)

    try:
        df = pd.read_csv(csv_path, header=None)
        npy_paths = [row.values[0] for _, row in df.iterrows()]
        if len(npy_paths) % 3 != 0:
            return render_template("batch.html", error="CSV length must be a multiple of 3 (ax, cor, sag per case).", device=str(DEVICE))

        # Prepare in-memory CSV
        out_buf = io.StringIO()
        writer = csv.writer(out_buf, lineterminator="\n")

        # group per case (ax, cor, sag)
        for i in range(0, len(npy_paths), 3):
            ax_path, co_path, sa_path = npy_paths[i:i+3]

            axial    = preprocess_data(ax_path, SLICE_TRANSFORM)
            coronal  = preprocess_data(co_path, SLICE_TRANSFORM)
            sagittal = preprocess_data(sa_path, SLICE_TRANSFORM)

            axial    = ensure_3ch(axial).unsqueeze(0).to(DEVICE)
            coronal  = ensure_3ch(coronal).unsqueeze(0).to(DEVICE)
            sagittal = ensure_3ch(sagittal).unsqueeze(0).to(DEVICE)

            probs = predict_case_from_arrays(axial, coronal, sagittal, MRNETS, LRS, device=DEVICE)
            writer.writerow([probs[0], probs[1], probs[2]])

        # Send as downloadable file
        out_buf.seek(0)
        return send_file(
            io.BytesIO(out_buf.getvalue().encode("utf-8")),
            mimetype="text/csv",
            as_attachment=True,
            download_name="predictions.csv",
        )
    except Exception as e:
        return render_template("batch.html", error=f"Batch prediction failed: {e}", device=str(DEVICE))


@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    print(f"⚙️ Device: {DEVICE}")
    app.run(debug=True)
