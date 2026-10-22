import sys
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "api"))

from medical_cnn.data import build_transforms


def test_transform_output_shape():
    image = Image.new("RGB", (300, 200))
    assert build_transforms(64)(image).shape == (3, 64, 64)


def test_transform_rejects_tiny_size():
    with pytest.raises(ValueError, match="image_size"):
        build_transforms(16)


def test_api_health_and_validation(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))

    # Import after setting ARTIFACTS_DIR so the API uses the isolated temporary directory.
    import main as api_main
    from fastapi.testclient import TestClient

    client = TestClient(api_main.app)
    assert client.get("/health").json()["model_artifacts_present"] is False

    buffer = BytesIO()
    Image.new("RGB", (32, 32)).save(buffer, format="PNG")
    response = client.post(
        "/api/predict",
        files={"file": ("x.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 503  # no trained model yet

    response = client.post(
        "/api/predict",
        files={"file": ("x.txt", b"hi", "text/plain")},
    )
    assert response.status_code == 415
