#!/usr/bin/env python3
import os, io, json, traceback
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as T
from huggingface_hub import hf_hub_download

# ─── Setup ──────────────────────────────────────────────────────────────
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from train_all_crops import build_backbone

# ─── Configuration ──────────────────────────────────────────────────────
REPO_ID = "Chaitanya412/crop-models"  # Hugging Face repo
CLASS_MAP_PATH = os.path.join(os.path.dirname(__file__), "class_map.json")

# load class‐map (list of labels)
with open(CLASS_MAP_PATH, "r") as f:
    CLASS_MAP = json.load(f)

# available model names
AVAILABLE = [
    "resnet50", "vgg16", "alexnet", "convnext_tiny", "vit_b_16", "mobilevit_s"
]

# preprocessing pipeline
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─── FastAPI Setup ──────────────────────────────────────────────────────
app = FastAPI(
    title="Crop‐Disease Classifier API",
    version="0.1.0",
    description="Upload a leaf image + choose model_name to get disease prediction"
)

# ─── Load Model from HuggingFace Hub ────────────────────────────────────
def load_model(name: str):
    if name not in AVAILABLE:
        raise HTTPException(400, f"Unknown model '{name}'. Choose from {AVAILABLE}")
    
    # Download the checkpoint file from HF hub
    ckpt_path = hf_hub_download(repo_id=REPO_ID, filename=f"best_all_{name}.pth")

    model = build_backbone(name, num_classes=len(CLASS_MAP), pretrained=False)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    return model.eval()

# ─── Routes ─────────────────────────────────────────────────────────────
@app.get("/ping")
def ping():
    return {"message": "API is working 🚀"}

@app.post("/predict/")
async def predict(
    model_name: str = Query(..., description="one of " + ", ".join(AVAILABLE)),
    file: UploadFile = File(..., description="Upload a leaf image file")
):
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "Uploaded file is not a valid image")

    x = preprocess(img).unsqueeze(0)

    try:
        model = load_model(model_name)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(probs.argmax().item())
            label = CLASS_MAP[idx]
            conf = float(probs[idx].item())
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Inference failed: {e}")

    return JSONResponse({
        "model": model_name,
        "predicted_label": label,
        "confidence_score": round(conf, 4)
    })

# ─── Serve HTML Frontend ────────────────────────────────────────────────
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
