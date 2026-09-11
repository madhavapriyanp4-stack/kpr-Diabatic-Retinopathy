"""
RetinaCheck backend API.

Run with:
    uvicorn main:app --reload --port 8000

Endpoints:
    POST /predict           -> upload a fundus image, get a DR grading
    GET  /patients           -> list all patients with their scan history
    GET  /patients/{id}      -> single patient detail
    GET  /overlays/{filename} -> serves the Grad-CAM overlay images
"""
import os

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from database import init_db, get_db, Patient, Scan
from predictor import get_predictor
from schemas import PatientOut, PredictResponse

OVERLAY_DIR = "storage/overlays"

app = FastAPI(title="RetinaCheck API")

# Allow the React dev server (typically localhost:5173 or :3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()
    get_predictor()  # loads the model once, up front, instead of on first request


os.makedirs(OVERLAY_DIR, exist_ok=True)
app.mount("/overlays", StaticFiles(directory=OVERLAY_DIR), name="overlays")


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
    patient_name: str = Form("Unregistered patient"),
    db: Session = Depends(get_db),
):
    image_bytes = await file.read()

    predictor = get_predictor()
    result = predictor.predict(image_bytes, overlay_dir=OVERLAY_DIR)

    # Create the patient record if this is their first scan
    patient = db.get(Patient, patient_id)
    if patient is None:
        patient = Patient(id=patient_id, name=patient_name)
        db.add(patient)
        db.commit()

    scan = Scan(
        patient_id=patient_id,
        stage_id=result["stage_id"],
        stage_name=result["stage_name"],
        confidence=result["confidence"],
        urgency=result["urgency"],
        recommended_action=result["recommended_action"],
        low_confidence_flag=result["low_confidence_flag"],
        original_filename=file.filename,
        overlay_filename=result["overlay_filename"],
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    return PredictResponse(
        scan_id=scan.id,
        patient_id=patient_id,
        stage_id=result["stage_id"],
        stage_name=result["stage_name"],
        confidence=result["confidence"],
        urgency=result["urgency"],
        recommended_action=result["recommended_action"],
        class_probabilities=result["class_probabilities"],
        low_confidence_flag=result["low_confidence_flag"],
        overlay_url=f"/overlays/{result['overlay_filename']}",
    )


@app.get("/patients", response_model=list[PatientOut])
def list_patients(db: Session = Depends(get_db)):
    return db.query(Patient).order_by(Patient.created_at.desc()).all()


@app.get("/patients/{patient_id}", response_model=PatientOut)
def get_patient(patient_id: str, db: Session = Depends(get_db)):
    patient = db.get(Patient, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@app.get("/health")
def health():
    return {"status": "ok"}
