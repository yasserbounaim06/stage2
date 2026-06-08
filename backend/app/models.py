import json
from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database import Base

class DatasetUpload(Base):
    __tablename__ = "dataset_uploads"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    filename = Column(String)
    storage_path = Column(String)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    is_validated = Column(Boolean, default=False)
    validation_message = Column(Text, nullable=True)
    num_images = Column(Integer, default=0)
    num_labels = Column(Integer, default=0)

    # Relationships
    training_jobs = relationship("TrainingJob", back_populates="dataset", cascade="all, delete-orphan")

class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("dataset_uploads.id"), nullable=False)
    status = Column(String, default="PENDING")  # PENDING, TRAINING, COMPLETED, FAILED
    epochs = Column(Integer, default=10)
    batch_size = Column(Integer, default=16)
    imgsz = Column(Integer, default=640)
    base_model = Column(String, default="yolov8n.pt")
    run_dir = Column(String, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    progress_percent = Column(Float, default=0.0)
    current_epoch = Column(Integer, default=0)
    metrics = Column(Text, nullable=True, default="{}")  # Stored as JSON string
    error_message = Column(Text, nullable=True)

    # Remote GPU Provider Tracking
    provider = Column(String, nullable=True)             # 'vastai' or 'salad'
    remote_job_id = Column(String, nullable=True)        # Remote instance/deployment ID
    remote_status = Column(String, nullable=True)        # Provider-reported lifecycle state
    training_logs = Column(Text, nullable=True)          # Saved worker execution outputs

    # Relationships
    dataset = relationship("DatasetUpload", back_populates="training_jobs")
    model_version = relationship("ModelVersion", uselist=False, back_populates="training_job", cascade="all, delete-orphan")

    def get_metrics(self):
        if not self.metrics:
            return {}
        try:
            return json.loads(self.metrics)
        except Exception:
            return {}

    def set_metrics(self, data):
        self.metrics = json.dumps(data)

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    training_job_id = Column(Integer, ForeignKey("training_jobs.id"), nullable=True)
    version_name = Column(String, unique=True, index=True)
    model_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=False)
    metrics = Column(Text, nullable=True, default="{}")  # Stored as JSON string

    # Relationships
    training_job = relationship("TrainingJob", back_populates="model_version")
    detections = relationship("DetectionResult", back_populates="model_version", cascade="all, delete-orphan")

    def get_metrics(self):
        if not self.metrics:
            return {}
        try:
            return json.loads(self.metrics)
        except Exception:
            return {}

    def set_metrics(self, data):
        self.metrics = json.dumps(data)

class DetectionResult(Base):
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False)
    filename = Column(String)
    image_path = Column(String)
    annotated_path = Column(String)
    detections = Column(Text, default="[]")  # Stored as JSON string list of bounding boxes
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    model_version = relationship("ModelVersion", back_populates="detections")

    def get_detections(self):
        if not self.detections:
            return []
        try:
            return json.loads(self.detections)
        except Exception:
            return []

    def set_detections(self, data):
        self.detections = json.dumps(data)
