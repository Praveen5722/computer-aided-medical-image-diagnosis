# MedVision AI — React frontend

A responsive React + Vite research interface for the **Computer-Aided Medical Image Diagnosis using Regularized Convolutional Neural Networks** project.

## Features

- Responsive medical imaging workspace and dashboard.
- Image upload by browsing or drag-and-drop; PNG, JPEG, WEBP and BMP, up to 15 MB.
- Image preview, scan-category selection and optional research reference.
- Connects to a real backend prediction endpoint; does not invent sample diagnoses when the model is not connected.
- Displays the returned prediction and optional confidence score; exports the response as JSON.
- Research-use disclaimer and de-identification reminders.

## Run locally

Requires Node.js 20.19+ or 22.12+.

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite.

## Connect the model API

Create `.env.local` inside this directory:

```env
VITE_API_URL=http://127.0.0.1:8000
```

The UI sends a multipart `POST` request to `{VITE_API_URL}/api/predict` with these fields:

- `file`: uploaded image
- `scan_type`: `chest-xray`, `brain-mri`, `skin-lesion`, or `other`
- `patient_id`: optional research reference

The frontend expects a JSON response with a prediction field (`prediction`, `predicted_class`, or `label`) and optionally `confidence` and `message`. Update `src/App.jsx` if your backend uses a different endpoint, field names, or response structure. Configure CORS on the API to allow the frontend origin. Never expose private keys in `VITE_*` variables.

Build for production:

```bash
npm run build
npm run preview
```

## Safety

This is a research interface, not a medical device or diagnostic service. It does not itself train or run the CNN. A real model API must be configured for inference. Do not upload identifiable patient data to an unapproved service, and do not use predictions as a substitute for qualified clinical review.
