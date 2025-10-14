import os
import io
import time
import random
from pathlib import Path
from datetime import datetime
from io import BytesIO
# Flask / Web
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# DB
import mysql.connector
from mysql.connector import Error
# PyTorch (add alongside other imports)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models
# ML / Imaging
from tensorflow import keras
from tensorflow.keras.utils import img_to_array
from PIL import Image, ImageOps, ImageFilter
import numpy as np

# OCR / PDF
import fitz  # PyMuPDF
import pytesseract
import google.generativeai as genai_old

# Gemini (NEW SDK) for Files API (blood report)
from google import genai as genai_new
from google.genai import types as gat
# Selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
_FASTAI_OK = True
try:
    from fastai.vision.all import (
        DataBlock, ImageBlock, CategoryBlock, Resize, aug_transforms,
        Normalize, imagenet_stats, vision_learner, PILImage
    )
except Exception:
    _FASTAI_OK = False
# =========================
# APP & GLOBAL CONFIG
# =========================
app = Flask(__name__)

# Secrets & config (set real env vars in production)
app.secret_key = os.environ.get('SECRET_KEY', 'super_secret_key')

db_config = {
    'host':     os.environ.get('DB_HOST', 'localhost'),
    'user':     os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('DB_PASSWORD', '624256'),
    'database': os.environ.get('DB_NAME', 'intellimedicare')
}

# Scan uploads (for /scan route)
SCAN_UPLOAD_FOLDER = os.path.join(app.static_folder or 'static', "uploads")
os.makedirs(SCAN_UPLOAD_FOLDER, exist_ok=True)

# Prescription uploads (temp storage for OCR files)
PRESC_UPLOAD_FOLDER = os.path.join(app.root_path, "prescription_uploads")
os.makedirs(PRESC_UPLOAD_FOLDER, exist_ok=True)
app.config["PRESC_UPLOAD_FOLDER"] = PRESC_UPLOAD_FOLDER

# Tesseract (allow override via env)
pytesseract.pytesseract.tesseract_cmd = os.environ.get(
    "TESSERACT_PATH",
    r"C:\Users\dhana\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)
TESS_LANG = os.environ.get("TESS_LANG", "eng")  # e.g., "eng+tam"
TESS_CONFIG = os.environ.get("TESS_CONFIG", "--oem 3 --psm 6")
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"}

# Gemini API (use env var in production)
API_KEY = os.environ.get("GOOGLE_API_KEY")

# New SDK (google.genai)
_gemini_client = genai_new.Client(api_key=API_KEY) if API_KEY else None

# Old SDK (google.generativeai) for legacy calls
if API_KEY:
    genai_old.configure(api_key=API_KEY)

# Selenium / ChromeDriver
CHROMEDRIVER_PATH = os.environ.get(
    "CHROME_DRIVER_PATH",
    r"C:\Users\dhana\Downloads\chromedriver-win64\chromedriver.exe"
)

\

# =========================
# DB HELPER
# =========================
def get_db_connection():
    return mysql.connector.connect(**db_config)

# =========================
# AUTH / CORE ROUTES
# =========================
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/signup', methods=['POST'])
def signup():
    email = request.form.get('email')
    full_name = request.form.get('fullName')
    username = request.form.get('username')
    password = generate_password_hash(request.form.get('password'))

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (email, full_name, username, password)
            VALUES (%s, %s, %s, %s)
        """, (email, full_name, username, password))
        conn.commit()
        flash("Signup successful! You can now log in.", "success")
    except mysql.connector.IntegrityError:
        flash("Email or username already exists.", "danger")
    except Error:
        flash("Database error occurred.", "danger")
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password_input = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row and check_password_hash(row[0], password_input):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            flash("Invalid credentials", "danger")

    return render_template('login_signup_ui.html')

@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', username=session['username'])

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ===== BLOOD REPORT (Gemini) CONFIG =====
BLOOD_ALLOWED_EXTS = {"pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"}
BLOOD_UPLOAD_FOLDER = os.path.join(app.root_path, "blood_uploads")
os.makedirs(BLOOD_UPLOAD_FOLDER, exist_ok=True)

# Prefer env var; falls back to placeholder
# Create client (ok to construct even if key missing; calls will fail gracefully)
_gemini_client = genai_new.Client(api_key=API_KEY)


# Model you want to use
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
def blood_allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in BLOOD_ALLOWED_EXTS

def blood_upload_to_gemini(file_path: str):
    # Works for PDFs and images
    return _gemini_client.files.upload(file=Path(file_path))

_BLOOD_SYSTEM_INSTRUCTION = (
    "You are a cautious clinical lab interpretation assistant. "
    "You are NOT a doctor and must NOT provide a diagnosis. "
    "Given ONLY the attached lab report file, identify up to five POSSIBLE medical "
    "conditions or nutrient deficiencies that could plausibly explain any abnormal values "
    "based strictly on the report's stated reference ranges (use age/sex-specific ranges if provided). "
    "If the report shows no clear abnormalities or lacks enough information, return an empty list."
)

_BLOOD_USER_PROMPT = (
    "Task: Based ONLY on the attached laboratory report file and its stated reference ranges, "
    "return a list of 0–5 POSSIBLE conditions or deficiencies that could explain clear abnormalities.\n"
    "Rules:\n"
    "- Do NOT diagnose.\n"
    "- Output only concise names (e.g., 'iron deficiency anemia', 'vitamin B12 deficiency', 'hypothyroidism', 'diabetes mellitus').\n"
    "- No numbers or explanations.\n"
    "- If insufficient evidence or normal, return an empty list.\n"
    "Output strictly as JSON: an array of strings."
)

def analyze_blood_report(file_path: str) -> list[str]:
    if not API_KEY or not _gemini_client:
        raise RuntimeError("GEMINI_API_KEY is not set on the server.")

    uploaded = blood_upload_to_gemini(file_path)

    resp = _gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[uploaded, _BLOOD_USER_PROMPT],
        config=gat.GenerateContentConfig(
            system_instruction=_BLOOD_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=list[str],  # force a JSON array of strings
        ),
    )

    items = []
    # Prefer parsed JSON
    if isinstance(getattr(resp, "parsed", None), list):
        items = [s.strip() for s in resp.parsed if isinstance(s, str) and s.strip()]
    else:
        # Fallback: parse text as JSON
        import json
        try:
            tmp = json.loads((resp.text or "[]").strip())
            if isinstance(tmp, list):
                items = [str(s).strip() for s in tmp if isinstance(s, str) and str(s).strip()]
        except Exception:
            items = []

    # Deduplicate case-insensitively; cap at 5
    seen, out = set(), []
    for s in items:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
        if len(out) >= 5:
            break
    return out


# =========================
# SCAN ANALYSIS
# =========================
# =========================
# SCAN ANALYSIS (Keras 3 SavedModel via TFSMLayer)
# =========================

# Path to your SavedModel dir (adjust if you use an env var)
SAVEDMODEL_DIR = os.environ.get("BRAIN_TUMOR_SAVEDMODEL", r"D:/Intellimedicare/Brain_tumor/model_saved")

# Wrap the SavedModel
tfsml = keras.layers.TFSMLayer(
    SAVEDMODEL_DIR,
    call_endpoint="serving_default"
)

# Match your SavedModel input signature shape/dtype
scan_inp = keras.Input(shape=(128, 128, 3), dtype="float32", name="input_2")

# IMPORTANT: select the correct output key from your signature (you used "dense_1")
scan_out = tfsml(scan_inp)["dense_1"]

scan_model = keras.Model(scan_inp, scan_out, name="wrapped_savedmodel")

class_labels = ['Glioma', 'Meningioma', 'notumor', 'Pituitary']

# Reuse the same uploads folder under static
SCAN_UPLOAD_FOLDER = os.path.join(app.static_folder or 'static', "uploads")
os.makedirs(SCAN_UPLOAD_FOLDER, exist_ok=True)
scan_inp = keras.Input(shape=(128, 128, 3), dtype="float32", name="input_2")

# IMPORTANT: select the correct output key from your signature (you used "dense_1")
scan_out = tfsml(scan_inp)["dense_1"]

scan_model = keras.Model(scan_inp, scan_out, name="wrapped_savedmodel")

class_labels = ['Glioma', 'Meningioma', 'notumor', 'Pituitary']

# Reuse the same uploads folder under static
SCAN_UPLOAD_FOLDER = os.path.join(app.static_folder or 'static', "uploads")
os.makedirs(SCAN_UPLOAD_FOLDER, exist_ok=True)

def predict_image(image):
    image = image.convert('RGB').resize((128, 128))
    img = img_to_array(image).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # (1,128,128,3)

    preds = scan_model.predict(img)

    preds = np.asarray(preds)
    if preds.ndim == 0:
        raise ValueError(f"Unexpected scalar prediction: {preds!r}")
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)
    elif preds.ndim > 2:
        preds = preds.reshape(preds.shape[0], -1)

    probs = preds[0]
    pred_index = int(np.argmax(probs))
    confidence = float(np.max(probs))

    label = class_labels[pred_index]
    result = "🟢 No Tumor Detected" if label == "notumor" else f"🔴 Tumor Detected: {label}"
    return result, f"{confidence*100:.2f}%"

# =========================
# SKIN CANCER (PyTorch) CONFIG
# =========================
SKIN_MODEL_PATH = os.environ.get("SKIN_MODEL_PATH", "D:\Intellimedicare\Skin_cancer\skincancer_model.pt")
SKIN_IMG_SIZE = int(os.environ.get("SKIN_IMG_SIZE", "112"))
SKIN_CLASS_NAMES = ["Benign", "Malignant"]

FORCE_CPU = os.getenv("FORCE_CPU", "0") == "1"
skin_device = torch.device("cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda:0")

skin_predict_transform = transforms.Compose([
    transforms.Resize((SKIN_IMG_SIZE, SKIN_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def _load_full_model_safely(path: str, device: torch.device):
    mdl = torch.load(path, map_location="cpu", weights_only=False)    # always map to CPU first
    if isinstance(mdl, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        mdl = mdl.module
    mdl.to(device)
    mdl.eval()
    return mdl

try:
    skin_model = _load_full_model_safely(SKIN_MODEL_PATH, skin_device)
    print(f"✅ Loaded skin-cancer model from {SKIN_MODEL_PATH} on device: {skin_device}")
except Exception as e:
    skin_model = None
    print(f"❌ Failed to load skin-cancer model from {SKIN_MODEL_PATH}: {e}")

def predict_skin_cancer(pil_img):
    """
    Returns (result_text, confidence_text, extra_info_dict)
    Uses either single-logit head or 2-logit head, auto-detected.
    """
    if skin_model is None:
        raise RuntimeError("Skin-cancer model not loaded. Check SKIN_MODEL_PATH.")

    with torch.no_grad():
        x = skin_predict_transform(pil_img.convert("RGB")).unsqueeze(0).to(skin_device, non_blocking=True)
        out = skin_model(x)

        # single-logit (B,1) or two-logit (B,2)
        if out.ndim == 2 and out.shape[-1] == 1:
            prob_malignant = torch.sigmoid(out).item()
            pred_idx = int(prob_malignant > 0.5)
            pred_label = SKIN_CLASS_NAMES[pred_idx]
            confidence = prob_malignant if pred_idx == 1 else (1 - prob_malignant)
        else:
            prob = F.softmax(out, dim=-1).squeeze(0)  # [2]
            conf, pred_idx_t = torch.max(prob, dim=-1)
            pred_idx = int(pred_idx_t.item())
            pred_label = SKIN_CLASS_NAMES[pred_idx]
            confidence = float(conf.item())

        result = f"{'🔴' if pred_label=='Malignant' else '🟢'} Skin Cancer: {pred_label}"
        return result, f"{confidence*100:.2f}%", {"device": str(skin_device), "pred_idx": pred_idx}
    
# =========================
# PNEUMONIA (X-ray) CONFIG
# =========================
PNEU_CLASSES = os.environ.get("PNEU_CLASSES", "Normal,Pneumonia").split(",")
PNEU_IMG_RESIZE = int(os.environ.get("PNEU_IMG_RESIZE", "460"))
PNEU_CROP_SIZE  = int(os.environ.get("PNEU_CROP_SIZE",  "224"))
PNEU_CKPT_PATH  = Path(os.environ.get("PNEU_CKPT_PATH", "D:\Intellimedicare\Pneumonia_Detection\pneumonia.pth")).resolve()

PNEU_IMAGENET_MEAN = [0.485, 0.456, 0.406]
PNEU_IMAGENET_STD  = [0.229, 0.224, 0.225]

pneu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def pneu_read_checkpoint(path: Path):
    return torch.load(path, map_location="cpu")

def pneu_looks_like_fastai(sd_like) -> bool:
    return isinstance(sd_like, dict) and "model" in sd_like and isinstance(sd_like["model"], dict)

def pneu_guess_resnet_from_keys(sd: dict) -> str:
    joined = " ".join(sd.keys())
    return "resnet50" if ".conv3." in joined else "resnet34"

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz: int = 1):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

def pneu_make_resnet_body_and_nf(arch_name: str):
    m = getattr(models, arch_name)(weights=None)
    body = nn.Sequential(*list(m.children())[:-2])
    nf = m.fc.in_features
    return body, nf

def pneu_build_fastai_like_model(arch_name: str, n_classes: int):
    body, nf = pneu_make_resnet_body_and_nf(arch_name)
    head = nn.Sequential(
        AdaptiveConcatPool2d(),
        nn.Flatten(),
        nn.BatchNorm1d(2*nf),
        nn.Dropout(0.5),
        nn.Linear(2*nf, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, n_classes),
    )
    return nn.Sequential(body, head)

def pneu_preprocess():
    return transforms.Compose([
        transforms.Resize(PNEU_IMG_RESIZE),
        transforms.CenterCrop(PNEU_CROP_SIZE),
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=PNEU_IMAGENET_MEAN, std=PNEU_IMAGENET_STD),
    ])

pneu_pre = pneu_preprocess()
pneu_use_fastai_runtime = False
pneu_learn = None
pneu_model = None

if PNEU_CKPT_PATH.exists():
    raw = pneu_read_checkpoint(PNEU_CKPT_PATH)
    if pneu_looks_like_fastai(raw):
        inner_sd = raw["model"]
        guessed_arch = os.environ.get("PNEU_ARCH") or pneu_guess_resnet_from_keys(inner_sd)
        if _FASTAI_OK:
            from fastai.vision.models import resnet18, resnet34, resnet50, resnet101, resnet152
            arch_map = {
                "resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50,
                "resnet101": resnet101, "resnet152": resnet152
            }
            if guessed_arch not in arch_map:
                guessed_arch = "resnet50"
            dblock = DataBlock(
                blocks=(ImageBlock, CategoryBlock(vocab=PNEU_CLASSES)),
                get_items=lambda _: [], get_y=lambda _: 0,
                item_tfms=[Resize(PNEU_IMG_RESIZE)],
                batch_tfms=[*aug_transforms(size=PNEU_CROP_SIZE), Normalize.from_stats(*imagenet_stats)],
            )
            dls = dblock.dataloaders(source=[], bs=1)
            pneu_learn = vision_learner(dls, arch_map[guessed_arch])
            pneu_learn.model_dir = PNEU_CKPT_PATH.parent
            pneu_learn.load(PNEU_CKPT_PATH.stem)
            pneu_learn.to(pneu_device).eval()
            pneu_use_fastai_runtime = True
        else:
            arch_name = os.environ.get("PNEU_ARCH") or pneu_guess_resnet_from_keys(inner_sd)
            pneu_model = pneu_build_fastai_like_model(arch_name, len(PNEU_CLASSES)).to(pneu_device)
            pneu_model.load_state_dict(inner_sd, strict=True)
            pneu_model.eval()
    else:
        sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        def strip_prefix(state_dict, prefix):
            keys = list(state_dict.keys())
            if keys and all(k.startswith(prefix) for k in keys):
                return {k[len(prefix):]: v for k, v in state_dict.items()}
            return state_dict
        for pfx in ("module.", "model."):
            sd = strip_prefix(sd, pfx)
        arch_name = os.environ.get("PNEU_ARCH", "resnet34")
        base = getattr(models, arch_name)(weights=None)
        base.fc = nn.Linear(base.fc.in_features, len(PNEU_CLASSES))
        pneu_model = base.to(pneu_device)
        pneu_model.load_state_dict(sd, strict=True)
        pneu_model.eval()
else:
    print(f"⚠️ Pneumonia checkpoint not found at {PNEU_CKPT_PATH}. Pneumonia mode will be disabled.")

def predict_pneumonia(pil_img):
    """
    Returns: (result_text, confidence_text, extra, dist) where:
      - result_text: "🟢 Normal" or "🔴 Pneumonia"
      - confidence_text: e.g., "97.12%"
      - extra: {"device": "..."}
      - dist: dict of class->prob (for optional UI bars)
    """
    if not (pneu_learn or pneu_model):
        raise RuntimeError("Pneumonia model not loaded. Provide PNEU_CKPT_PATH.")

    if pneu_use_fastai_runtime:
        img = PILImage.create(BytesIO(pil_img.tobytes()))  # ensure stream-like
        with torch.no_grad():
            pred_class, pred_idx, probs = pneu_learn.predict(img)
        pairs = [(str(c), float(probs[i])) for i, c in enumerate(pneu_learn.dls.vocab)]
    else:
        x = pneu_pre(pil_img).unsqueeze(0).to(pneu_device)
        with torch.no_grad():
            logits = pneu_model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pairs = list(zip(PNEU_CLASSES, [float(p) for p in probs]))

    pairs.sort(key=lambda kv: kv[1], reverse=True)
    top_class, top_prob = pairs[0]

    result = f"{'🔴 Pneumonia' if top_class.lower()=='pneumonia' else '🟢 Normal'}"
    return result, f"{top_prob*100:.2f}%", {"device": str(pneu_device)}, dict(pairs)


@app.route('/scan', methods=['GET', 'POST'])
def scan():
    if 'username' not in session:
        return redirect(url_for('login'))

    result = None
    confidence = None
    image_file = None
    scan_type = request.form.get('scan_type', 'brain')  # default to brain
    extra = {}  # for optional details (e.g., device)

    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename:
            try:
                from werkzeug.utils import secure_filename
                safe_name = secure_filename(f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
                save_path = os.path.join(SCAN_UPLOAD_FOLDER, safe_name)

                img = Image.open(file.stream).convert('RGB')

                if scan_type == 'skin':
                    result, confidence, extra = predict_skin_cancer(img)
                elif scan_type == 'pneumonia':
                    result, confidence, extra, _dist = predict_pneumonia(img)
                else:
                    # brain tumor (existing TF/Keras flow)
                    result, confidence = predict_image(img)

                # save uploaded
                img.save(save_path, quality=92)
                image_file = f"uploads/{safe_name}"
            except Exception as e:
                flash(f"Image handling error: {str(e)}", "danger")
        else:
            flash("Please choose an image file.", "warning")

    return render_template('scan.html',
                           result=result,
                           confidence=confidence,
                           image_file=image_file,
                           scan_type=scan_type,
                           extra=extra)


# =========================
# BLOOD REPORT (placeholder)
# =========================
@app.route('/blood-report', methods=['GET', 'POST'])
def blood_report():
    if 'username' not in session:
        return redirect(url_for('login'))

    data: list[str] = []
    error: str | None = None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            error = "No file selected."
        elif not blood_allowed_file(file.filename):
            error = "Invalid file type. Upload a PDF or image (png/jpg/jpeg/tiff/bmp/webp)."
        else:
            safe_name = secure_filename(file.filename)
            file_path = os.path.join(BLOOD_UPLOAD_FOLDER, safe_name)
            file.save(file_path)
            try:
                issues = analyze_blood_report(file_path)
                if issues:
                    data = issues
                else:
                    error = "No clear issues found based on the report’s reference ranges."
            except Exception as e:
                error = f"Processing failed: {e}"
            finally:
                try:
                    os.remove(file_path)
                except Exception:
                    pass

    # Renders templates/blood_report.html (see minimal template below)
    return render_template('blood_report.html', data=data, error=error)


# =========================================================
# PRESCRIPTION MODULE (OCR + GEMINI + 1MG SCRAPER)
# =========================================================

# ---- OCR helpers ----
def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    img = pil_img.convert("L")
    if min(img.size) < 1000:
        scale = 1.5
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    return img

def ocr_pil_image(pil_img: Image.Image) -> str:
    prepped = preprocess_for_ocr(pil_img)
    text = pytesseract.image_to_string(prepped, lang=TESS_LANG, config=TESS_CONFIG)
    return (text or "").strip()

# ---- Extractors ----
def extract_text_from_pdf(pdf_path: str) -> str:
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Could not open PDF: {e}")

    parts = []
    for i, page in enumerate(doc):
        text = (page.get_text("text") or "").strip()
        if not text:
            pix = page.get_pixmap(dpi=300, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = ocr_pil_image(img)
        if text:
            parts.append(f"\n--- Page {i+1} ---\n{text}")
    doc.close()
    return "\n".join(parts).strip()

def extract_text_from_image(img_path: str) -> str:
    p = Path(img_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    try:
        texts = []
        with Image.open(p) as im:
            i = 0
            while True:
                try:
                    im.seek(i)
                    page_img = im.copy()
                    page_text = ocr_pil_image(page_img)
                    if page_text:
                        texts.append(f"\n--- Image Page {i+1} ---\n{page_text}")
                    i += 1
                except EOFError:
                    break
        return "\n".join(texts).strip()
    except Exception as e:
        raise RuntimeError(f"Could not open image: {e}")

# ---- Gemini helpers ----
def parse_bullets_to_list(raw: str) -> list:
    meds = []
    for line in (raw or "").splitlines():
        s = line.strip()
        if not s:
            continue
        for prefix in ["- ", "* ", "• ", "· ", "— ", "-- "]:
            if s.startswith(prefix):
                s = s[len(prefix):].strip()
                break
        if len(s) > 2 and (s[0].isdigit() and (s[1] in {")", "."})):
            s = s[2:].strip()
        s = s.rstrip(" .;:,")
        if s:
            meds.append(s)
    seen = set()
    out = []
    for m in meds:
        if m.lower() not in seen:
            seen.add(m.lower())
            out.append(m)
    return out

def get_prescribed_medicines(extracted_text: str) -> list:
    if not API_KEY:
        # Fallback: naive heuristic if API key not set
        lines = [ln.strip() for ln in extracted_text.splitlines() if ln.strip()]
        rough = [ln for ln in lines if len(ln.split()) <= 4]
        return parse_bullets_to_list("\n".join(rough))

    # Old SDK is already configured globally when API_KEY is present
    model = genai_old.GenerativeModel(model_name="gemini-1.5-flash")
    prompt = f"""
You are a medical prescription analyzer. Extract the list of prescribed medicines from the following prescription text.
- Output only the medicines in a simple bullet list (one per line).
- Do not include dosage, route, or instructions.
- If no medicines are found, reply with exactly: No medicines prescribed.

Prescription text:
{extracted_text}
"""
    resp = model.generate_content(prompt)
    text = (getattr(resp, "text", "") or "").strip()
    if text == "No medicines prescribed.":
        return []
    return parse_bullets_to_list(text)


# ---- Wrapper ----
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_medicines(file_path: str) -> list:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        extracted_text = extract_text_from_pdf(file_path)
    elif ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        extracted_text = extract_text_from_image(file_path)
    else:
        raise ValueError("Unsupported file type. Use PDF or image (jpg, png, tiff, bmp, webp).")
    if not extracted_text:
        return []
    return get_prescribed_medicines(extracted_text)

# ---- 1mg Scraper (single, shared) ----
def scrape_1mg(medicine_name: str) -> dict:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")

    driver = webdriver.Chrome(service=Service(CHROMEDRIVER_PATH), options=options)
    driver.set_page_load_timeout(30)

    result = {"Medicine Name": medicine_name}
    search_url = f"https://www.1mg.com/search/all?name={medicine_name}"

    try:
        driver.get(search_url)
        WebDriverWait(driver, 12).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a[href^='/drugs/'], a[href^='/otc/']"))
        )
        time.sleep(random.uniform(1.2, 2.0))
        items = driver.find_elements(By.CSS_SELECTOR, "a[href^='/drugs/'], a[href^='/otc/']")
        href = None
        for item in items:
            link = item.get_attribute("href")
            if link:
                href = link
                break
        if not href:
            result["error"] = "No product found."
            return result

        driver.get(href)
        time.sleep(random.uniform(1.0, 1.8))

        result["Brand Name"] = driver.find_element(By.CSS_SELECTOR, ".DrugHeader__title-content___2ZaPo").text if driver.find_elements(By.CSS_SELECTOR, ".DrugHeader__title-content___2ZaPo") else "N/A"
        result["Manufacturer"] = driver.find_element(By.XPATH, "//div[contains(text(),'Marketer')]/following-sibling::div").text if driver.find_elements(By.XPATH, "//div[contains(text(),'Marketer')]/following-sibling::div") else "N/A"
        result["Salt"] = driver.find_element(By.XPATH, "//div[contains(.,'SALT COMPOSITION')]/following-sibling::div").text if driver.find_elements(By.XPATH, "//div[contains(.,'SALT COMPOSITION')]/following-sibling::div") else "N/A"
        price_el = (driver.find_elements(By.CSS_SELECTOR, "span.PriceBoxPlanOption__offer-price___3v9x8") or
                    driver.find_elements(By.CSS_SELECTOR, "[class*='offer-price']") or
                    driver.find_elements(By.CSS_SELECTOR, "[data-testid='price']"))
        result["Price"] = price_el[0].text if price_el else "N/A"

        uses = [li.text for li in driver.find_elements(By.CSS_SELECTOR, "#uses_and_benefits li")]
        result["Uses"] = ", ".join(uses) if uses else "N/A"

        side_effects = [li.text for li in driver.find_elements(By.CSS_SELECTOR, "#side_effects ul li")]
        result["Side Effects"] = ", ".join(side_effects) if side_effects else "N/A"

        result["URL"] = href

    except (TimeoutException, WebDriverException) as e:
        result["error"] = str(e)
    finally:
        driver.quit()

    return result

# ---- Route: /prescription ----
@app.route('/prescription', methods=['GET', 'POST'])
def prescription():
    if 'username' not in session:
        return redirect(url_for('login'))

    data = []       # list of dicts from scrape_1mg
    error = None    # error message to show in template
    extracted = []  # list of extracted medicine names (for display)

    if request.method == "POST":
        # 1) Manual single medicine query (optional)
        med_name = (request.form.get('medicine') or '').strip()
        if med_name:
            data.append(scrape_1mg(med_name))

        # 2) File upload OCR → Gemini → list of meds (optional)
        file = request.files.get('file')
        if file and file.filename:
            if allowed_file(file.filename):
                filename = secure_filename(f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
                file_path = os.path.join(app.config["PRESC_UPLOAD_FOLDER"], filename)
                file.save(file_path)
                try:
                    extracted = extract_medicines(file_path)
                    if not extracted and not med_name:
                        error = "No medicines extracted from the file."
                    else:
                        for med in extracted:
                            data.append(scrape_1mg(med))
                except Exception as e:
                    error = f"Extraction error: {e}"
                finally:
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
            else:
                error = "Invalid file type. Use PDF or image."
        elif not med_name and not file:
            error = "Provide a medicine name or upload a prescription file."

    return render_template('prescription.html', data=data, error=error, extracted=extracted)

# =========================
# MAIN
# =========================
if __name__ == '__main__':
    # Consider host='0.0.0.0' when containerized
    app.run(debug=True)
