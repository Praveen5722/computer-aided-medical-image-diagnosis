# Computer-Aided Medical Image Diagnosis using Regularized CNNs

An educational research prototype for training a regularized convolutional neural network (CNN) to classify medical images. Includes training and evaluation scripts, a Streamlit image-upload interface, a React frontend workspace, and unit tests.

> **Safety:** This project is not a medical device and has not been clinically validated. Do not use it to diagnose, treat, or rule out disease. Medical decisions belong with qualified clinicians.

## Features
- PyTorch CNN with batch normalization, dropout, and weight decay.
- Training from class-organized image folders, validation, early stopping, and checkpoint saving.
- Evaluation with accuracy, precision, recall, F1, confusion matrix, and classification report.
- Streamlit interface for uploading an image and reviewing model scores.
- **React + Vite frontend** with responsive dashboard, image drag-and-drop, preview, scan category selection, API integration, and JSON response export.
- Unit tests and architecture documentation.

## Project structure
```text
frontend/           React + Vite frontend workspace
  src/App.jsx       Image analysis interface and API integration
  src/styles.css    Responsive UI styling
  README.md         Frontend setup and backend API contract
medical_cnn/        Model, transforms, checkpoint helpers
scripts/train.py    Train the CNN
scripts/evaluate.py Evaluate on a held-out dataset
tests/              Unit tests
docs/               Architecture notes
app.py              Streamlit demo
requirements.txt    Python dependencies
```

## React frontend

See [frontend/README.md](frontend/README.md) for setup and API configuration.

```bash
cd frontend
npm install
npm run dev
```

To enable predictions, configure `VITE_API_URL` in `frontend/.env.local` to point to a backend exposing `POST /api/predict`. The frontend does not generate placeholder diagnoses when the API is unavailable. If your existing Python app uses a different route or request/response schema, update the integration in `frontend/src/App.jsx` or add a compatible API endpoint. See the frontend README for the expected multipart fields and response format.

## 1. Python setup
Python 3.10 or 3.11 recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Prepare data
The project uses torchvision ImageFolder format. Use de-identified, properly licensed images and split by patient (not individual images) to prevent leakage.

```text
data/medical_images/
├── train/
│   ├── class_a/
│   └── class_b/
├── val/
│   ├── class_a/
│   └── class_b/
└── test/
    ├── class_a/
    └── class_b/
```

Replace class names with labels from your dataset. Dataset and pretrained weights are not included. Keep private patient data out of Git.

## 3. Train
```bash
python scripts/train.py --data data/medical_images --epochs 25 --batch-size 16 --image-size 224
```
Best checkpoint, class names, and training history are written to `artifacts/`.

## 4. Evaluate
```bash
python scripts/evaluate.py --data data/medical_images/test --checkpoint artifacts/best_model.pt
```
Do not use the test set to tune hyperparameters.

## 5. Launch the Streamlit interface
```bash
streamlit run app.py
```
Train a model first. The app expects `artifacts/best_model.pt` and `artifacts/classes.json`.

## 6. Run tests
```bash
pytest -q
```

## Regularization
- Batch normalization stabilizes intermediate activations.
- Dropout reduces co-adaptation of features.
- AdamW weight decay discourages excessively large weights.
- Small training-only augmentations can improve robustness, but must be appropriate for the modality and task. Flips may change laterality and should not be used automatically.

## Recommended research practice
Use patient-level splits, a locked held-out test set, external validation, class-wise sensitivity/specificity, confidence intervals, calibration, subgroup analysis, and expert review of failure cases. Accuracy alone is not evidence of clinical utility.

## Limitations
This starter does not include data, pretrained weights, DICOM handling, external validation, calibration, or clinical workflow integration. It is for education and research only. The React frontend is a separate interface and requires a compatible prediction API for live model inference.


## 7. Run with Docker Compose

Install Docker Desktop (Windows/macOS) or Docker Engine with the Compose plugin (Linux). Train the model first so that `artifacts/best_model.pt` and `artifacts/classes.json` exist locally.

```bash
docker compose up --build
```

Then open:
- React frontend: http://localhost:8080
- API health: http://localhost:8000/health (API is internal in Compose; use the frontend's same-origin `/api` proxy, or publish port 8000 temporarily for local debugging).

The Compose setup builds the React frontend with Nginx and a FastAPI inference service. Model artifacts are mounted read-only and are not baked into the images. The original Streamlit demo is optional:

```bash
docker compose --profile streamlit up --build
```

It will be available at http://localhost:8501.

To stop the services, run `docker compose down`. The inference API returns HTTP 503 until the trained checkpoint and class list are present. Do not commit datasets, trained model files, or identifiable patient images to Git.
