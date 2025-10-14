import os
from pathlib import Path
from io import BytesIO

from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# Try to import fastai (optional). If not present, we'll do a pure-PyTorch rebuild.
_FASTAI_OK = True
try:
    from fastai.vision.all import (
        DataBlock, ImageBlock, CategoryBlock, Resize, aug_transforms,
        Normalize, imagenet_stats, vision_learner, PILImage
    )
except Exception:
    _FASTAI_OK = False

# =========================
# CONFIG — EDIT IF NEEDED
# =========================
# Default: look for "pneumonia.pth" in the same folder as this file.
DEFAULT_CKPT = Path(__file__).resolve().with_name("pneumonia.pth")
CKPT_PATH  = Path(os.environ.get("CKPT_PATH", str(DEFAULT_CKPT)))

# MUST match your training label order:
CLASSES    = os.environ.get("CLASSES", "Normal,Pneumonia").split(",")

# Inference sizes (match training if you changed them)
IMG_RESIZE = int(os.environ.get("IMG_RESIZE", "460"))     # pre-crop resize
CROP_SIZE  = int(os.environ.get("CROP_SIZE", "224"))      # final crop

# ImageNet stats (typical)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# HELPERS
# =========================
def read_checkpoint(path: Path):
    # Keep weights_only=False because FastAI checkpoints include non-tensor objects (optimizer, etc.).
    # The warning is fine as long as you trust your file.
    return torch.load(path, map_location="cpu")

def looks_like_fastai(sd_like) -> bool:
    # FastAI learn.save(): {'model': <state_dict>, 'opt': ...}
    if isinstance(sd_like, dict) and "model" in sd_like and isinstance(sd_like["model"], dict):
        return True
    return False

def guess_resnet_from_keys(sd: dict) -> str:
    # Bottleneck resnets (50/101/152) have conv1/conv2/conv3 per block; BasicBlocks (18/34) have only conv1/conv2.
    joined = " ".join(sd.keys())
    return "resnet50" if ".conv3." in joined else "resnet34"

class AdaptiveConcatPool2d(nn.Module):
    """FastAI-style: concat AdaptiveAvgPool2d and AdaptiveMaxPool2d along channel dim."""
    def __init__(self, sz: int = 1):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

def make_resnet_body_and_nf(arch_name: str):
    # Build torchvision resnet and return the "body" (up to layer4) + num_features for head.
    if not hasattr(models, arch_name):
        raise ValueError(f"Unknown torchvision model '{arch_name}'.")
    m = getattr(models, arch_name)(weights=None)
    # Body = all children except avgpool and fc (i.e., keep conv-bn-relu-maxpool-layer1..layer4)
    body = nn.Sequential(*list(m.children())[:-2])
    nf = m.fc.in_features  # features that feed the original fc
    return body, nf 

def build_fastai_like_model(arch_name: str, n_classes: int):
    """
    Recreate the typical FastAI cnn_learner structure in pure PyTorch:
      model = nn.Sequential( body, head )
    where head is: AdaptiveConcatPool2d -> Flatten -> BN -> Dropout -> Linear -> ReLU -> BN -> Dropout -> Linear
    """
    body, nf = make_resnet_body_and_nf(arch_name)
    head = nn.Sequential(
        AdaptiveConcatPool2d(),                 # [B, 2*nf, 1, 1]
        nn.Flatten(),                           # [B, 2*nf]
        nn.BatchNorm1d(2*nf),
        nn.Dropout(p=0.5),
        nn.Linear(2*nf, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.5),
        nn.Linear(512, n_classes)
    )
    return nn.Sequential(body, head)

def pytorch_preprocess():
    return transforms.Compose([
        transforms.Resize(IMG_RESIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# =========================
# LOAD MODEL
# =========================
preprocess = pytorch_preprocess()
use_fastai_runtime = False
learn = None
model = None

if not CKPT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}\n"
                            f"Tip: put 'pneumonia.pth' next to app.py, or set CKPT_PATH env var.")

raw = read_checkpoint(CKPT_PATH)

if looks_like_fastai(raw):
    # FastAI path (preferred if fastai is installed)
    inner_sd = raw["model"]  # FastAI: {'model': state_dict, 'opt': ...}
    guessed_arch = os.environ.get("ARCH") or guess_resnet_from_keys(inner_sd)

    if _FASTAI_OK:
        # Build a dummy learner with your vocab so head size matches exactly
        from fastai.vision.models import resnet18, resnet34, resnet50, resnet101, resnet152
        arch_map = {
            "resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50,
            "resnet101": resnet101, "resnet152": resnet152
        }
        if guessed_arch not in arch_map:
            guessed_arch = "resnet50"  # sensible default for bottleneck keys

        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock(vocab=CLASSES)),
            get_items=lambda _: [],
            get_y=lambda _: 0,
            item_tfms=[Resize(IMG_RESIZE)],
            batch_tfms=[*aug_transforms(size=CROP_SIZE), Normalize.from_stats(*imagenet_stats)]
        )
        dls = dblock.dataloaders(source=[], bs=1)
        learn = vision_learner(dls, arch_map[guessed_arch])
        learn.model_dir = CKPT_PATH.parent
        # load by stem name (e.g., "pneumonia" for "pneumonia.pth")
        learn.load(CKPT_PATH.stem)
        learn.to(device).eval()
        use_fastai_runtime = True
    else:
        # Pure PyTorch rebuild of FastAI cnn_learner: Sequential(body, head)
        arch_name = os.environ.get("ARCH") or guess_resnet_from_keys(inner_sd)
        model = build_fastai_like_model(arch_name, len(CLASSES)).to(device)
        # STRICT load of the exact FastAI 'model' state dict with '0.' and '1.' keys
        missing, unexpected = model.load_state_dict(inner_sd, strict=True)
        model.eval()
else:
    # Raw state_dict path (not likely your case, but supported)
    sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    def strip_prefix(state_dict, prefix):
        keys = list(state_dict.keys())
        if keys and all(k.startswith(prefix) for k in keys):
            return {k[len(prefix):]: v for k, v in state_dict.items()}
        return state_dict
    for pfx in ("module.", "model."):
        sd = strip_prefix(sd, pfx)

    arch_name = os.environ.get("ARCH", "resnet34")
    # Standard torchvision model (avgpool+fc head)
    base = getattr(models, arch_name)(weights=None)
    in_features = base.fc.in_features
    base.fc = nn.Linear(in_features, len(CLASSES))
    model = base.to(device)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    model.eval()

# =========================
# FLASK
# =========================
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or request.files["file"].filename == "":
        return redirect(url_for("index"))
    f = request.files["file"]

    if use_fastai_runtime: 
        img = PILImage.create(f.stream)
        with torch.no_grad():
            pred_class, pred_idx, probs = learn.predict(img)
        pairs = [(str(c), float(probs[i])) for i, c in enumerate(learn.dls.vocab)]
        pairs.sort(key=lambda kv: kv[1], reverse=True)
        top_class, top_prob = pairs[0]
    else:
        img = Image.open(BytesIO(f.read()))
        x = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        pairs = [(cls, float(probs[i])) for i, cls in enumerate(CLASSES)]
        pairs.sort(key=lambda kv: kv[1], reverse=True)
        top_class, top_prob = pairs[0]

    return render_template(
        "index.html",
        top_class=top_class,
        top_prob=top_prob,
        result={k: v for k, v in pairs}
    )

if __name__ == "__main__":
    # Windows example to override arch if needed:
    #   set ARCH=resnet50 && python app.py         (CMD)
    #   $env:ARCH="resnet50"; python app.py        (PowerShell)
    app.run(host="0.0.0.0", port=5000, debug=False)
