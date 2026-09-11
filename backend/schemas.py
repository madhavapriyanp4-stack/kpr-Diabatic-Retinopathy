"""Pydantic response models -- defines the exact JSON shape the API returns."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ScanOut(BaseModel):
    id: int
    stage_id: int
    stage_name: str
    confidence: float
    urgency: str
    recommended_action: str
    low_confidence_flag: bool
    overlay_filename: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class PatientOut(BaseModel):
    id: str
    name: str
    age: Optional[int]
    sex: Optional[str]
    created_at: datetime
    scans: list[ScanOut] = []

    class Config:
        from_attributes = True


class PredictResponse(BaseModel):
    scan_id: int
    patient_id: str
    stage_id: int
    stage_name: str
    confidence: float
    urgency: str
    recommended_action: str
    class_probabilities: dict
    low_confidence_flag: bool
    overlay_url: str
