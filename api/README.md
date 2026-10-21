# CNN inference API

The FastAPI service loads the repository's trained model from `artifacts/best_model.pt` and `artifacts/classes.json`.

- `GET /health`: service and artifact status.
- `POST /api/predict`: multipart fields `file`, `scan_type`, and optional `patient_id`.

The API does not persist uploaded images or the optional research reference. Use only de-identified research images. Run behind HTTPS with appropriate access controls before any non-local use.

The model artifacts are intentionally excluded from Docker build contexts and Git. Train the model first and keep the two artifact files in the local `artifacts/` directory. A service can start without them, but prediction requests return HTTP 503 until both are present.
