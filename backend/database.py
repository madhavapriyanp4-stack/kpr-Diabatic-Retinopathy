"""
SQLite database for patients and their scan history, via SQLAlchemy.
Creates dr_screening.db in the working directory on first run.
"""
from datetime import datetime

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DATABASE_URL = "sqlite:///./dr_screening.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Patient(Base):
    __tablename__ = "patients"

    id = Column(String, primary_key=True)          # e.g. "PT-1042"
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=True)
    sex = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    scans = relationship("Scan", back_populates="patient", order_by="Scan.created_at")


class Scan(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, ForeignKey("patients.id"))

    stage_id = Column(Integer, nullable=False)
    stage_name = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    urgency = Column(String, nullable=False)
    recommended_action = Column(String, nullable=False)
    low_confidence_flag = Column(Boolean, default=False)

    original_filename = Column(String, nullable=True)
    overlay_filename = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="scans")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
