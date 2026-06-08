from datetime import datetime
from typing import List, Dict, Any, Optional
import json
from pydantic import BaseModel, field_validator

# Dataset
class DatasetUploadBase(BaseModel):
    name: str

class DatasetUploadResponse(DatasetUploadBase):
    id: int
    filename: str
    uploaded_at: datetime
    is_validated: bool
    validation_message: Optional[str]
    num_images: int
    num_labels: int

    class Config:
        from_attributes = True

# Training Job
class TrainingStartRequest(BaseModel):
    dataset_id: int
    epochs: Optional[int] = 10
    batch_size: Optional[int] = 16
    imgsz: Optional[int] = 640
    base_model: Optional[str] = "yolov8n.pt"

class TrainingJobResponse(BaseModel):
    id: int
    dataset_id: int
    status: str
    epochs: int
    batch_size: int
    imgsz: int
    base_model: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_percent: float
    current_epoch: int
    metrics: Dict[str, Any]
    error_message: Optional[str] = None
    provider: Optional[str] = None
    remote_job_id: Optional[str] = None

    @field_validator('metrics', mode='before')
    @classmethod
    def parse_metrics(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return {}
        return v

    class Config:
        from_attributes = True

# Model Version
class ModelVersionResponse(BaseModel):
    id: int
    training_job_id: Optional[int]
    version_name: str
    created_at: datetime
    is_active: bool
    metrics: Dict[str, Any]

    @field_validator('metrics', mode='before')
    @classmethod
    def parse_metrics(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return {}
        return v

    class Config:
        from_attributes = True

# Detection Result
class DetectionItem(BaseModel):
    class_name: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2] relative or absolute
    text: Optional[str] = None  # Extracted container number if applicable

class DetectionResultResponse(BaseModel):
    id: int
    model_version_id: int
    filename: str
    image_url: str
    annotated_url: str
    detections: List[DetectionItem]
    created_at: datetime

    class Config:
        from_attributes = True

# Dashboard Stats
class DashboardStats(BaseModel):
    total_datasets: int
    total_models: int
    latest_job: Optional[TrainingJobResponse] = None
    latest_inference: Optional[DetectionResultResponse] = None

# Run Details (incorporating Dataset and Model info)
class RunDetailsResponse(BaseModel):
    job: TrainingJobResponse
    dataset_name: str
    model_version_name: Optional[str] = None

    class Config:
        from_attributes = True
