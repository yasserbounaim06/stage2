import os
import sys
from typing import List, Optional
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from pathlib import Path
from datetime import datetime

# Path helper to support running from stage2 root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import UPLOAD_DIR, INFERENCE_DIR, MODEL_DIR, DEFAULT_BASE_MODEL
from app.database import engine, Base, get_db
from app.models import DatasetUpload, TrainingJob, ModelVersion, DetectionResult
from app.schemas import (
    DatasetUploadResponse, TrainingStartRequest, TrainingJobResponse,
    ModelVersionResponse, DetectionResultResponse, DashboardStats, RunDetailsResponse, DetectionItem
)
from app.services.storage_service import StorageService
from app.services.dataset_service import DatasetService
from app.services.training_service import training_service
from app.services.inference_service import InferenceService
from app.services.remote_training_service import remote_training_manager

# 1. Initialize SQLite Database Tables
Base.metadata.create_all(bind=engine)

def run_migrations():
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns("training_jobs")]
    
    with engine.begin() as conn:
        if "provider" not in columns:
            conn.execute(text("ALTER TABLE training_jobs ADD COLUMN provider VARCHAR"))
            print("Migration: Added provider column to training_jobs.")
        if "remote_job_id" not in columns:
            conn.execute(text("ALTER TABLE training_jobs ADD COLUMN remote_job_id VARCHAR"))
            print("Migration: Added remote_job_id column.")
        if "remote_status" not in columns:
            conn.execute(text("ALTER TABLE training_jobs ADD COLUMN remote_status VARCHAR"))
            print("Migration: Added remote_status column.")
        if "training_logs" not in columns:
            conn.execute(text("ALTER TABLE training_jobs ADD COLUMN training_logs TEXT"))
            print("Migration: Added training_logs column.")

run_migrations()


# 2. Initialize FastAPI
app = FastAPI(
    title="Custom YOLO Training and Container Number Detection API",
    description="Backend services for fine-tuning YOLO models on custom datasets and executing OCR inferences.",
    version="1.0.0"
)

# 3. Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development ease
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Mount Static Directories for serving files
# Make sure dirs exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/static/inference", StaticFiles(directory=str(INFERENCE_DIR)), name="inference")

# 5. API Routes

@app.post("/api/datasets/upload", response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Uploads a training dataset (ZIP archive) and validates its YOLO format structure.
    """
    # Verify extension
    suffix = Path(file.filename).suffix.lower()
    if suffix != ".zip":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only ZIP archives are supported."
        )
        
    # Create DB entry to get a unique ID
    db_dataset = DatasetUpload(name=name, filename=file.filename)
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    
    try:
        # Save ZIP
        file_bytes = await file.read()
        zip_path = StorageService.save_upload(file_bytes, file.filename, db_dataset.id)
        
        # Extract ZIP
        extracted_dir = StorageService.get_extracted_dir(db_dataset.id)
        StorageService.extract_zip(zip_path, extracted_dir)
        
        # Run validation
        validation_results = DatasetService.validate_dataset(extracted_dir)
        
        # Update DB entry
        db_dataset.is_validated = validation_results["valid"]
        db_dataset.validation_message = validation_results["message"]
        db_dataset.num_images = validation_results["num_images"]
        db_dataset.num_labels = validation_results["num_labels"]
        # Update path to where it was extracted
        db_dataset.storage_path = str(StorageService.get_dataset_dir(db_dataset.id).absolute())
        
        db.commit()
        db.refresh(db_dataset)
        
        return db_dataset
        
    except Exception as e:
        # Cleanup and delete DB entry on failure
        db.delete(db_dataset)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while saving/validating the dataset: {str(e)}"
        )

@app.get("/api/datasets", response_model=List[DatasetUploadResponse])
def list_datasets(db: Session = Depends(get_db)):
    """
    Lists all uploaded datasets.
    """
    datasets = db.query(DatasetUpload).order_by(DatasetUpload.uploaded_at.desc()).all()
    return datasets

@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """
    Deletes a dataset, cleans up files on disk, and cascade-deletes associated training runs.
    """
    dataset = db.query(DatasetUpload).filter(DatasetUpload.id == dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {dataset_id} not found."
        )
        
    try:
        # Find all model versions associated with this dataset's training jobs
        jobs = db.query(TrainingJob).filter(TrainingJob.dataset_id == dataset_id).all()
        for job in jobs:
            if job.model_version:
                model_path = Path(job.model_version.model_path)
                if model_path.exists() and model_path.is_file():
                    try:
                        model_path.unlink()
                        print(f"Deleted model weights file: {model_path}")
                    except Exception as e:
                        print(f"Error deleting model weights file: {e}")
                        
        # Delete dataset files on disk
        StorageService.clean_dataset_dir(dataset_id)
        
        # Delete from database (cascades to jobs, model versions, etc.)
        db.delete(dataset)
        db.commit()
        return {"message": f"Dataset {dataset_id} ('{dataset.name}') and all associated training jobs / models were deleted."}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete dataset: {str(e)}"
        )

@app.post("/api/training/start", response_model=TrainingJobResponse, status_code=status.HTTP_202_ACCEPTED)
def start_training(payload: TrainingStartRequest, db: Session = Depends(get_db)):
    """
    Registers a training job and puts it in the background execution queue.
    """
    # Verify dataset exists and is validated
    dataset = db.query(DatasetUpload).filter(DatasetUpload.id == payload.dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {payload.dataset_id} not found."
        )
        
    if not dataset.is_validated:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot start training on an unvalidated dataset. Please fix validation errors first."
        )
        
    # Register the job
    job = TrainingJob(
        dataset_id=payload.dataset_id,
        epochs=payload.epochs,
        batch_size=payload.batch_size,
        imgsz=payload.imgsz,
        base_model=payload.base_model or DEFAULT_BASE_MODEL,
        status="PENDING"
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Enqueue in local training manager
    training_service.enqueue_job(job.id)
    
    return job

@app.get("/api/training/status/{job_id}", response_model=TrainingJobResponse)
def get_training_status(job_id: int, db: Session = Depends(get_db)):
    """
    Gets real-time training progress details (status, epoch history, loss rates, metrics).
    """
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job with ID {job_id} not found."
        )
    return job

@app.get("/api/runs/{run_id}", response_model=RunDetailsResponse)
def get_run_details(run_id: int, db: Session = Depends(get_db)):
    """
    Gets extended details for a training run.
    """
    job = db.query(TrainingJob).filter(TrainingJob.id == run_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training run with ID {run_id} not found."
        )
        
    dataset = db.query(DatasetUpload).filter(DatasetUpload.id == job.dataset_id).first()
    dataset_name = dataset.name if dataset else "Unknown Dataset"
    
    model_version = db.query(ModelVersion).filter(ModelVersion.training_job_id == run_id).first()
    model_version_name = model_version.version_name if model_version else None
    
    return RunDetailsResponse(
        job=job,
        dataset_name=dataset_name,
        model_version_name=model_version_name
    )

@app.get("/api/models", response_model=List[ModelVersionResponse])
def get_models(db: Session = Depends(get_db)):
    """
    Lists all trained model versions.
    """
    models = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
    return models

@app.post("/api/models/{model_id}/activate")
def activate_model(model_id: int, db: Session = Depends(get_db)):
    """
    Sets a trained model version as the active model for inferences.
    """
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version {model_id} not found."
        )
        
    # Deactivate all models
    db.query(ModelVersion).update({ModelVersion.is_active: False})
    
    # Activate selected model
    model.is_active = True
    db.commit()
    return {"message": f"Model '{model.version_name}' is now set as the active model."}

@app.delete("/api/models/{model_id}")
def delete_model(model_id: int, db: Session = Depends(get_db)):
    """
    Deletes a registered model version and cleans up the weight file on disk.
    """
    model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version with ID {model_id} not found."
        )
        
    was_active = model.is_active
    
    try:
        # Delete file on disk
        model_path = Path(model.model_path)
        if model_path.exists() and model_path.is_file():
            try:
                model_path.unlink()
                print(f"Deleted model weights file: {model_path}")
            except Exception as e:
                print(f"Error deleting model weights file: {e}")
                
        db.delete(model)
        db.commit()
        
        if was_active:
            # Try to activate the default base model version or another existing version
            remaining_model = db.query(ModelVersion).order_by(ModelVersion.id.desc()).first()
            if remaining_model:
                remaining_model.is_active = True
                db.commit()
                print(f"Activated next available model: {remaining_model.version_name}")
                
        return {"message": f"Model version {model_id} ('{model.version_name}') was successfully deleted."}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model version: {str(e)}"
        )

@app.post("/api/inference/detect", response_model=DetectionResultResponse)
async def run_detection(
    model_id: Optional[int] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Uploads a test image, runs inference using the selected (or active default) YOLO model,
    renders bounding boxes, crops text targets, runs OCR text extraction, and returns JSON.
    """
    # 1. Fetch active model or requested model
    if model_id:
        model_ver = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
        if not model_ver:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Requested model version {model_id} not found."
            )
    else:
        # Find default active model
        model_ver = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
        
    # Use base model filename if no trained version is registered yet
    model_path = model_ver.model_path if model_ver else DEFAULT_BASE_MODEL
    model_version_id = model_ver.id if model_ver else 0
    
    # Create a dummy model record in database if running off standard base model
    if model_version_id == 0:
        # See if default model version exists
        base_ver = db.query(ModelVersion).filter(ModelVersion.version_name == "Base_yolov8n").first()
        if not base_ver:
            base_ver = ModelVersion(
                version_name="Base_yolov8n",
                model_path=DEFAULT_BASE_MODEL,
                is_active=True,
                metrics={"description": "Default pre-trained COCO base model"}
            )
            db.add(base_ver)
            db.commit()
            db.refresh(base_ver)
        model_version_id = base_ver.id
        model_path = DEFAULT_BASE_MODEL

    # 2. Save test image
    file_bytes = await file.read()
    file_path, relative_web_path = StorageService.save_inference_image(file_bytes, file.filename)
    
    try:
        # 3. Run YOLO inference & OCR
        detection_data = InferenceService.run_detection(model_path, file_path)
        
        # 4. Save detection result in DB
        # Format detection items
        db_detections = []
        for det in detection_data["detections"]:
            db_detections.append(
                DetectionItem(
                    class_name=det["class_name"],
                    confidence=det["conf"],
                    box=det["box"],
                    text=det.get("text", "")
                )
            )
            
        result = DetectionResult(
            model_version_id=model_version_id,
            filename=file.filename,
            image_path=str(relative_web_path.as_posix()),
            annotated_path=str(Path(detection_data["annotated_url"]).relative_to("/static").as_posix())
        )
        result.set_detections([d.dict() for d in db_detections])
        db.add(result)
        db.commit()
        db.refresh(result)
        
        # 5. Format response
        response_data = DetectionResultResponse(
            id=result.id,
            model_version_id=result.model_version_id,
            filename=result.filename,
            image_url=f"/static/{result.image_path}",
            annotated_url=f"/static/{result.annotated_path}",
            detections=db_detections,
            created_at=result.created_at
        )
        
        return response_data
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(e)}"
        )

@app.get("/api/dashboard/stats", response_model=DashboardStats)
def get_dashboard_stats(db: Session = Depends(get_db)):
    """
    Returns high-level statistics for the dashboard visualization.
    """
    total_datasets = db.query(DatasetUpload).count()
    total_models = db.query(ModelVersion).count()
    
    # Latest training job
    latest_job = db.query(TrainingJob).order_by(TrainingJob.id.desc()).first()
    
    # Latest detection result
    latest_inf_result = db.query(DetectionResult).order_by(DetectionResult.id.desc()).first()
    
    latest_inference = None
    if latest_inf_result:
        # Fetch detection items
        items = [
            DetectionItem(**item) for item in latest_inf_result.get_detections()
        ]
        latest_inference = DetectionResultResponse(
            id=latest_inf_result.id,
            model_version_id=latest_inf_result.model_version_id,
            filename=latest_inf_result.filename,
            image_url=f"/static/{latest_inf_result.image_path}",
            annotated_url=f"/static/{latest_inf_result.annotated_path}",
            detections=items,
            created_at=latest_inf_result.created_at
        )
        
    return DashboardStats(
        total_datasets=total_datasets,
        total_models=total_models,
        latest_job=latest_job,
        latest_inference=latest_inference
    )
