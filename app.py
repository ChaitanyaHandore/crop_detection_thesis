#!/usr/bin/env python3
import os, io, json, traceback
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as T

# ensure our training‐script is importable
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from train_all_crops import build_backbone

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(__file__)
MODELS_DIR     = os.path.join(BASE_DIR, "models")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "class_map.json")

# load class‐map (list of labels)
with open(CLASS_MAP_PATH, "r") as f:
    CLASS_MAP = json.load(f)


# map model_name → checkpoint file
AVAILABLE = {
    name: os.path.join(MODELS_DIR, f"best_all_{name}.pth")
    for name in ["resnet50","vgg16","alexnet","convnext_tiny","vit_b_16","mobilevit_s"]
}

# preprocessing pipeline
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


# ─── FastAPI setup ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Crop‐Disease Classifier API",
    version="0.1.0",
    description="Upload a leaf image + choose model_name to get disease prediction"
)

def load_model(name: str):
    """
    Build the backbone, load its .pth, and return eval‐mode model.
    """
    if name not in AVAILABLE:
        raise HTTPException(400, f"Unknown model '{name}'. Choose from {list(AVAILABLE)}")
    ckpt = AVAILABLE[name]
    if not os.path.isfile(ckpt):
        raise HTTPException(500, f"Checkpoint not found: {ckpt}")
    model = build_backbone(name, num_classes=len(CLASS_MAP), pretrained=False)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    return model.eval()

@app.get("/")
def index():
    return {"message":"Crop‐Disease Classifier API"}

@app.post("/predict/")
async def predict(
    model_name: str = Query(..., description="one of " + ", ".join(AVAILABLE.keys())),
    file: UploadFile = File(..., description="Upload a leaf image file")
):
    # 1) Read & validate image
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "Uploaded file is not a valid image")

    # 2) Preprocess
    x = preprocess(img).unsqueeze(0)  # add batch dim

    # 3) Inference
    try:
        model = load_model(model_name)
        with torch.no_grad():
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)[0]
            idx    = int(probs.argmax().item())
            label  = CLASS_MAP[idx]
            conf   = float(probs[idx].item())
    except Exception as e:
        traceback.print_exc()  # print full stack to your terminal
        raise HTTPException(500, f"Inference failed: {e}")

    # 4) Return JSON
    return JSONResponse({
        "model":            model_name,
        "predicted_label":  label,
        "confidence_score": round(conf, 4)
    })