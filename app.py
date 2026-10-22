from pathlib import Path
import json
import streamlit as st
import torch
from PIL import Image
from medical_cnn.data import build_transforms, load_checkpoint
from medical_cnn.model import RegularizedCNN

ROOT = Path(__file__).parent
CHECKPOINT = ROOT / "artifacts" / "best_model.pt"
CLASSES = ROOT / "artifacts" / "classes.json"

st.set_page_config(page_title="Medical Image CNN Research Demo", page_icon="🩻", layout="centered")
st.title("Computer-Aided Medical Image Diagnosis")
st.caption("Regularized Convolutional Neural Network — research prototype")
st.warning("Educational/research use only. This model is not clinically validated and must not be used for diagnosis or treatment decisions.")
if not CHECKPOINT.exists() or not CLASSES.exists():
    st.info("No trained model found. Prepare a dataset and run scripts/train.py first.")
    st.stop()

@st.cache_resource
def load_model_bundle(checkpoint_path: str, classes_path: str):
    """Load model weights once and reuse the model across Streamlit reruns."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(Path(checkpoint_path), device)
    classes = json.loads(Path(classes_path).read_text(encoding="utf-8"))
    model = RegularizedCNN(checkpoint["num_classes"], checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    transform = build_transforms(checkpoint["image_size"])
    return model, classes, transform, device


model, classes, transform, device = load_model_bundle(str(CHECKPOINT), str(CLASSES))
uploaded = st.file_uploader("Upload a medical image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image")
        with torch.inference_mode():
            probabilities = torch.softmax(model(transform(image).unsqueeze(0).to(device)), dim=1)[0].cpu()
        ranked = sorted(zip(classes, probabilities.tolist()), key=lambda item: item[1], reverse=True)
        st.subheader("Model output (not a diagnosis)")
        st.write(f"Highest-scoring class: **{ranked[0][0]}**")
        for label, probability in ranked:
            st.write(f"{label}: {probability:.1%}")
            st.progress(float(probability))
        st.caption("Scores are model outputs and may not be calibrated measures of clinical certainty.")
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
