# RetinaCheck Backend

FastAPI service that wraps the trained DR classifier for the RetinaCheck
frontend. Handles image upload, inference (stage + confidence + Grad-CAM),
and patient/scan history storage.

## Required folder layout

This backend imports directly from `dr_model/`, so they need to sit
side by side:

```
project/
  dr_model/        <- the model code (already have this)
  backend/         <- this folder
```

## Setup

```bash
pip install -r backend/requirements.txt
```

Put your trained checkpoint (`best_model.pt`) in this `backend/` folder,
or point to it elsewhere with an environment variable:

```bash
export DR_CHECKPOINT_PATH=/path/to/best_model.pt
```

## Run

```bash
uvicorn backend.main:app --reload --port 8000
```

First startup will take a few seconds while the model loads. Once
running, visit `http://localhost:8000/docs` for interactive API docs
(FastAPI auto-generates this).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | Upload a fundus image + patient info, get DR grading |
| GET | `/patients` | List all patients with scan history |
| GET | `/patients/{id}` | Single patient's full history |
| GET | `/overlays/{filename}` | Serves the Grad-CAM overlay image |
| GET | `/health` | Simple liveness check |

### Example: calling /predict with curl

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@scan.png" \
  -F "patient_id=PT-1042" \
  -F "patient_name=R. Meena"
```

Returns:
```json
{
  "scan_id": 1,
  "patient_id": "PT-1042",
  "stage_id": 2,
  "stage_name": "Moderate NPDR",
  "confidence": 84.2,
  "urgency": "Review",
  "recommended_action": "Review in 3-6 months",
  "class_probabilities": {...},
  "low_confidence_flag": false,
  "overlay_url": "/overlays/a1b2c3d4.png"
}
```

## Data storage

- `dr_screening.db` -- SQLite file, created automatically on first run.
  Holds patients and scan records.
- `storage/overlays/` -- Grad-CAM overlay images, created automatically.

Both are local files. On Render, the default filesystem is ephemeral, so
records and overlays can disappear after a redeploy or service restart.
Use a Render persistent disk (paid service) or change `DATABASE_URL` to a
managed PostgreSQL database and move overlays to object storage for durable
production data.

## Deploying to Render

The repository includes `render.yaml`, which defines separate backend and
frontend services.

1. Push the repository to GitHub and create a new Render Blueprint from it.
2. Deploy the `retinacheck-api` service first. Wait for `/health` to return
  `{"status":"ok"}` and copy its public URL.
3. On `retinacheck-frontend`, set `VITE_API_BASE_URL` to the backend URL,
  without a trailing slash, for example `https://retinacheck-api.onrender.com`.
4. Deploy the frontend and copy its public URL.
5. On `retinacheck-api`, set `CORS_ORIGINS` to the frontend URL, without a
  trailing slash, then redeploy the backend.
6. Open the frontend URL, upload a scan, and verify that the result overlay
  loads. The API docs are available at `<backend-url>/docs`.

The model is loaded during backend startup. If the service runs out of memory,
select a larger Render instance; CPU inference is supported, but each scan may
take longer.

## Connecting the React frontend

In the frontend, replace the mock `handleAnalyze` function's random
result generation with a real call:

```javascript
const formData = new FormData();
formData.append("file", imageFile);
formData.append("patient_id", patientId);
formData.append("patient_name", patientName);

const res = await fetch("http://localhost:8000/predict", {
  method: "POST",
  body: formData,
});
const result = await res.json();
```

The response fields (`stage_id`, `confidence`, `overlay_url`, etc.) map
directly onto what the result screen already expects -- just swap the
data source, the UI shouldn't need structural changes.
