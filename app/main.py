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
  :root { --bg:#0f172a; --card:#1e293b; --line:#334155; --text:#e2e8f0; --muted:#94a3b8; --accent:#38bdf8; --ok:#34d399; }
  * { box-sizing:border-box; }
  body { margin:0; min-height:100vh; padding:32px 16px; color:var(--text);
         background:radial-gradient(circle at 15% -10%, #1e40af55, transparent 45%), var(--bg);
         font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
         display:flex; justify-content:center; }
  .wrap { width:100%; max-width:760px; }
  h1 { margin:0 0 6px; font-size:28px; }
  .sub { margin:0 0 24px; color:var(--muted); line-height:1.5; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:24px; }
  .drop { display:block; border:2px dashed #475569; border-radius:12px; padding:28px; text-align:center;
          cursor:pointer; transition:border-color .2s, background .2s; }
  .drop:hover, .drop.drag { border-color:var(--accent); background:#0ea5e911; }
  .drop img { max-width:100%; max-height:300px; border-radius:8px; margin-top:14px; display:none; }
  .hint { color:var(--muted); font-size:14px; margin-top:6px; }
  button { width:100%; margin-top:16px; padding:13px; font-size:16px; font-weight:600; color:#0f172a;
           background:var(--accent); border:0; border-radius:10px; cursor:pointer; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  .result { display:none; margin-top:24px; }
  .result.show { display:block; }
  .pred { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }
  .pred .label { font-size:32px; font-weight:700; color:var(--ok); }
  .pred .conf { color:var(--muted); }
  .bars { display:grid; gap:8px; margin-top:16px; }
  .bar { display:grid; grid-template-columns:120px 1fr 56px; gap:10px; align-items:center; font-size:14px; }
  .track { height:10px; background:var(--line); border-radius:6px; overflow:hidden; }
  .fill { height:100%; width:0; background:linear-gradient(90deg,var(--accent),var(--ok)); border-radius:6px;
          transition:width .5s ease; }
  .pct { text-align:right; color:var(--muted); }
  .err { display:none; margin-top:16px; color:#f87171; }
  .err.show { display:block; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Road Surface Condition Classifier</h1>
  <p class="sub">上传一张路面照片，模型会判断路面状态：dry / wet / water / fresh_snow / melted_snow / ice。</p>
  <div class="card">
    <label class="drop" id="drop">
      <input type="file" id="file" accept="image/*" hidden>
      <div><b>点击选择图片</b>，或把图片拖到这里</div>
      <div class="hint">支持 JPG / PNG</div>
      <img id="preview" alt="">
    </label>
    <button id="go" disabled>Predict</button>
    <div class="err" id="err"></div>
    <div class="result" id="result">
      <div class="pred">
        <span class="label" id="label"></span>
        <span class="conf" id="conf"></span>
      </div>
      <div class="bars" id="bars"></div>
    </div>
  </div>
</div>
<script>
var fileInput = document.getElementById('file');
var drop = document.getElementById('drop');
var preview = document.getElementById('preview');
var go = document.getElementById('go');
var err = document.getElementById('err');
var result = document.getElementById('result');
var label = document.getElementById('label');
var conf = document.getElementById('conf');
var bars = document.getElementById('bars');

function showError(msg) { err.textContent = msg; err.classList.add('show'); }
function clearError() { err.classList.remove('show'); }

fileInput.addEventListener('change', function () {
  var f = fileInput.files[0];
  if (!f) return;
  preview.src = URL.createObjectURL(f);
  preview.style.display = 'block';
  go.disabled = false;
});

drop.addEventListener('dragover', function (e) { e.preventDefault(); drop.classList.add('drag'); });
drop.addEventListener('dragleave', function () { drop.classList.remove('drag'); });
drop.addEventListener('drop', function (e) {
  e.preventDefault();
  drop.classList.remove('drag');
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    fileInput.dispatchEvent(new Event('change'));
  }
});

go.addEventListener('click', async function () {
  var f = fileInput.files[0];
  if (!f) return;
  go.disabled = true;
  go.textContent = 'Predicting...';
  clearError();
  try {
    var fd = new FormData();
    fd.append('file', f);
    var r = await fetch('/predict', { method: 'POST', body: fd });
    var j = await r.json();
    if (!r.ok) { throw new Error(j.detail || ('HTTP ' + r.status)); }
    label.textContent = j.condition;
    conf.textContent = '(conf ' + (j.confidence * 100).toFixed(1) + '%)';
    bars.innerHTML = '';
    var entries = Object.entries(j.probabilities).sort(function (a, b) { return b[1] - a[1]; });
    entries.forEach(function (entry) {
      var row = document.createElement('div');
      row.className = 'bar';
      row.innerHTML = '<span>' + entry[0] + '</span>' +
        '<div class="track"><div class="fill"></div></div>' +
        '<span class="pct">' + (entry[1] * 100).toFixed(1) + '%</span>';
      bars.appendChild(row);
      requestAnimationFrame(function () {
        row.querySelector('.fill').style.width = (entry[1] * 100) + '%';
      });
    });
    result.classList.add('show');
  } catch (e) {
    showError('Prediction failed: ' + e.message);
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
