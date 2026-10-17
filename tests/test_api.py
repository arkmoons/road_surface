"""API tests for the road-surface classifier.

The real model weights are NOT required: ``get_model`` is monkeypatched with a
deterministic dummy model, so these tests check the *HTTP contract* and edge
cases and run fast on CI without any downloaded data or weights.
"""

import io

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_mod


class DummyModel(nn.Module):
    """Always predicts the same 6 logits, independent of the input image."""

    def forward(self, x):
        logits = torch.tensor([1.0, 2.0, 3.0, 0.5, 0.2, 0.1])
        return logits.unsqueeze(0).expand(x.size(0), -1)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(main_mod, "get_model", lambda: DummyModel())
    return TestClient(main_mod.app)


def _image_bytes(fmt: str = "PNG", size=(240, 360)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(120, 120, 120)).save(buf, format=fmt)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #
def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --------------------------------------------------------------------------- #
# /predict - happy path
# --------------------------------------------------------------------------- #
def test_predict_returns_valid_condition_and_probabilities(client):
    resp = client.post("/predict", files={"file": ("road.png", _image_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["condition"] in main_mod.CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    probs = body["probabilities"]
    assert set(probs) == set(main_mod.CLASSES)
    assert abs(sum(probs.values()) - 1.0) < 1e-5


def test_predict_accepts_jpeg(client):
    resp = client.post("/predict", files={"file": ("road.jpg", _image_bytes("JPEG"), "image/jpeg")})
    assert resp.status_code == 200
    assert resp.json()["condition"] in main_mod.CLASSES


# --------------------------------------------------------------------------- #
# /predict - edge cases / invalid input
# --------------------------------------------------------------------------- #
def test_predict_rejects_non_image_content_type(client):
    resp = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 400


def test_predict_rejects_corrupt_image_bytes(client):
    # valid-looking content type, but the bytes are not a real image
    resp = client.post("/predict", files={"file": ("broken.png", b"not an image", "image/png")})
    assert resp.status_code == 400


def test_predict_requires_a_file(client):
    resp = client.post("/predict")
    assert resp.status_code == 422
