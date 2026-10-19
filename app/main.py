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
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Road Surface Classifier</title>
<style>
  body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
         max-width: 640px; margin: 48px auto; padding: 0 16px; color: #1f2937; background: #f8fafc; }
  h1 { font-size: 24px; margin: 0 0 6px; }
  p { color: #6b7280; margin: 0 0 24px; }
  .box { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 20px; }
  input[type=file] { margin-bottom: 12px; }
  button { padding: 8px 16px; font-size: 15px; color: #fff; background: #2563eb;
           border: 0; border-radius: 6px; cursor: pointer; }
  button:disabled { background: #9ca3af; cursor: not-allowed; }
  img { display: none; max-width: 100%; max-height: 220px; margin-top: 12px; border-radius: 6px; }
  .result { display: none; margin-top: 16px; }
  .result.show { display: block; }
  .result .line { font-size: 18px; }
  .result b { color: #16a34a; }
  ul { margin: 8px 0 0; padding-left: 18px; color: #4b5563; font-size: 14px; }
</style>
</head>
<body>
<h1>Road Surface Condition Classifier</h1>
<p>Upload a road photo; the model returns one of: dry, wet, water, fresh_snow, melted_snow, ice.</p>
<div class="box">
  <input type="file" id="file" accept="image/*">
  <button id="go" disabled>Predict</button>
  <img id="preview" alt="">
  <div class="result" id="result">
    <div class="line">Prediction: <b id="label"></b> <span id="conf"></span></div>
    <ul id="probs"></ul>
  </div>
</div>
<script>
var file = document.getElementById('file');
var go = document.getElementById('go');
var preview = document.getElementById('preview');
var result = document.getElementById('result');
var label = document.getElementById('label');
var conf = document.getElementById('conf');
var probs = document.getElementById('probs');

file.addEventListener('change', function () {
  var f = file.files[0];
  if (!f) return;
  preview.src = URL.createObjectURL(f);
  preview.style.display = 'block';
  go.disabled = false;
});

go.addEventListener('click', async function () {
  var f = file.files[0];
  if (!f) return;
  go.disabled = true;
  go.textContent = 'Predicting...';
  try {
    var fd = new FormData();
    fd.append('file', f);
    var r = await fetch('/predict', { method: 'POST', body: fd });
    var j = await r.json();
    if (!r.ok) { throw new Error(j.detail || ('HTTP ' + r.status)); }
    label.textContent = j.condition;
    conf.textContent = '(' + (j.confidence * 100).toFixed(1) + '%)';
    probs.innerHTML = '';
    var entries = Object.entries(j.probabilities).sort(function (a, b) { return b[1] - a[1]; });
    entries.forEach(function (e) {
      var li = document.createElement('li');
      li.textContent = e[0] + ': ' + (e[1] * 100).toFixed(1) + '%';
      probs.appendChild(li);
    });
    result.classList.add('show');
  } catch (e) {
    alert('Prediction failed: ' + e.message);
  } finally {
    go.disabled = false;
    go.textContent = 'Predict';
  }
});
</script>
</body>
</html>"""


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
