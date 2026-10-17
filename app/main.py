"""FastAPI service that serves the road-surface-condition classifier.

The served model is read from ``ml/registry/current_model.md`` (written by
``ml/train.py``), falling back to the ``MODEL_BACKBONE`` / ``MODEL_PATH``
environment variables.
"""

import io
import os
import re
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from torchvision import models, transforms

CLASSES = ["dry", "wet", "water", "fresh_snow", "melted_snow", "ice"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

REGISTRY_FILE = Path(__file__).resolve().parent.parent / "ml" / "registry" / "current_model.md"


def build_model(backbone: str, num_classes: int = len(CLASSES)) -> nn.Module:
    if backbone == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif backbone == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"unknown backbone: {backbone}")
    return model


def resolve_model() -> tuple[str, str]:
    """Return (backbone, weights_path) for the production model."""
    backbone = os.environ.get("MODEL_BACKBONE", "resnet18")
    weights = os.environ.get("MODEL_PATH", "models/current.pt")
    if REGISTRY_FILE.exists():
        text = REGISTRY_FILE.read_text(encoding="utf-8")
        m_back = re.search(r"backbone:\s*(\S+)", text)
        m_path = re.search(r"weights:\s*(\S+)", text)
        if m_back:
            backbone = m_back.group(1)
        if m_path:
            weights = m_path.group(1)
    return backbone, weights


_MODEL = None


def get_model() -> nn.Module:
    global _MODEL
    if _MODEL is None:
        backbone, weights = resolve_model()
        model = build_model(backbone)
        state = torch.load(weights, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _MODEL = model
    return _MODEL


_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


def predict_image(pil_image: Image.Image) -> dict:
    model = get_model()
    img = pil_image.convert("RGB")
    tensor = _TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    idx = int(probs.argmax())
    return {
        "condition": CLASSES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {cls: float(p) for cls, p in zip(CLASSES, probs)},
    }


app = FastAPI(title="Road Surface Condition Classifier", version="1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Road Surface Classifier</title></head>
<body style="font-family:sans-serif;max-width:640px;margin:40px auto">
  <h1>Road Surface Condition Classifier</h1>
  <p>Upload a road image; the model returns one of:
     dry, wet, water, fresh_snow, melted_snow, ice.</p>
  <input type="file" id="file" accept="image/*">
  <button onclick="predict()">Predict</button>
  <pre id="out"></pre>
  <script>
  async function predict() {
    const f = document.getElementById('file').files[0];
    if (!f) return;
    const fd = new FormData(); fd.append('file', f);
    const r = await fetch('/predict', {method:'POST', body: fd});
    const j = await r.json();
    document.getElementById('out').textContent =
      j.condition + '  (' + (j.confidence*100).toFixed(1) + '%)\\n' +
      JSON.stringify(j.probabilities, null, 2);
  }
  </script>
</body></html>"""


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
    return JSONResponse(predict_image(img))
