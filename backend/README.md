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
cd backend
pip install -r requirements.txt
```

Put your trained checkpoint (`best_model.pt`) in this `backend/` folder,
or point to it elsewhere with an environment variable:

```bash
export DR_CHECKPOINT_PATH=/path/to/best_model.pt
```

## Run

```bash
uvicorn main:app --reload --port 8000
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

Both are local files; fine for a demo/prototype, not meant for
production multi-user deployment as-is.

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
