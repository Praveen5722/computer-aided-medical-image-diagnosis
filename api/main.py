"""Minimal FastAPI inference endpoint for the repository's regularized CNN."""
import json
import os
from io import BytesIO
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from medical_cnn.data import build_transforms, load_checkpoint
from medical_cnn.model import RegularizedCNN

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = Path(os.getenv("ARTIFACTS_DIR", str(ROOT / "artifacts")))
CHECKPOINT_PATH = ARTIFACTS / "best_model.pt"
CLASSES_PATH = ARTIFACTS / "classes.json"
MAX_IMAGE_BYTES = 15 * 1024 * 1024
ALLOWED_ORIGINS = [v.strip() for v in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8080").split(",") if v.strip()]

app = FastAPI(title="Medical CNN Research API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_classes = None
_transform = None
_model_lock = Lock()


def load_model():
    global _model, _classes, _transform
    if _model is not None:
        return
    if not CHECKPOINT_PATH.is_file() or not CLASSES_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Trained model not found. Train the CNN and provide artifacts/best_model.pt and artifacts/classes.json.",
        )
    try:
        checkpoint = load_checkpoint(CHECKPOINT_PATH, device)
        classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
        if not isinstance(classes, list) or len(classes) != checkpoint["num_classes"]:
            raise ValueError("classes.json must be a list matching the checkpoint's num_classes.")
        model = RegularizedCNN(checkpoint["num_classes"], checkpoint["dropout"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _model = model
        _classes = classes
        _transform = build_transforms(checkpoint["image_size"])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load model artifacts: {exc}") from exc


@app.get("/health")
def health():
    ready = CHECKPOINT_PATH.is_file() and CLASSES_PATH.is_file()
    return {"status": "ok", "model_artifacts_present": ready, "device": str(device)}


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    scan_type: str = Form("other"),
    patient_id: str | None = Form(None),
):
    # The optional research reference is intentionally not persisted.
    del patient_id
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a supported image file.")
    raw = await file.read(MAX_IMAGE_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 15 MB limit.")
    try:
        with Image.open(BytesIO(raw)) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="File is not a valid readable image.") from exc
    with _model_lock:
        load_model()
        try:
            tensor = _transform(image).unsqueeze(0).to(device)
            with torch.inference_mode():
                probabilities = torch.softmax(_model(tensor), dim=1)[0].cpu().tolist()
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Image could not be processed: {exc}") from exc
    ranked = sorted(
        ({"label": str(label), "confidence": float(score)} for label, score in zip(_classes, probabilities)),
        key=lambda item: item["confidence"],
        reverse=True,
    )
    return {
        "prediction": ranked[0]["label"],
        "predicted_class": ranked[0]["label"],
        "confidence": ranked[0]["confidence"],
        "probabilities": ranked,
        "scan_type": scan_type,
        "message": "Experimental model output only; not a clinical diagnosis. Confidence scores may not be calibrated.",
    }
