import pytest
import torch
from PIL import Image

import app.main as main_mod


def test_class_list_is_the_six_conditions():
    assert main_mod.CLASSES == ["dry", "wet", "water", "fresh_snow", "melted_snow", "ice"]


def test_build_model_outputs_one_logit_per_class():
    model = main_mod.build_model("resnet18")
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 6)


def test_build_model_rejects_unknown_backbone():
    with pytest.raises(ValueError):
        main_mod.build_model("totally-made-up")


def test_preprocess_produces_expected_tensor_shape():
    img = Image.new("RGB", (240, 360), color=(0, 0, 0))
    tensor = main_mod._TRANSFORM(img)
    assert tensor.shape == (3, 224, 224)


def test_preprocess_accepts_grayscale_and_converts_to_rgb():
    gray = Image.new("L", (100, 100), color=77)
    assert gray.convert("RGB").size == (100, 100)
