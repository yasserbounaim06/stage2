import queue
import threading
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from app.config import YOLO_RUNS_DIR, MODEL_DIR, DEFAULT_BASE_MODEL, MOCK_TRAINING
from app.database import SessionLocal
from app.models import TrainingJob, ModelVersion, DatasetUpload

class TrainingService:
    def __init__(self):
        self.queue = queue.Queue()
        self.current_job_id = None
        self.worker_thread = None
        self.lock = threading.Lock()
        
    def enqueue_job(self, job_id: int):
        """Adds a job ID to the queue and starts the worker thread if idle."""
        self.queue.put(job_id)
        self.start_worker()

    def start_worker(self):
        """Starts the background worker thread if it's not already running."""
        with self.lock:
            if self.worker_thread is None or not self.worker_thread.is_alive():
                self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
                self.worker_thread.start()

    def get_queue_status(self) -> dict:
        """Returns details about the active job and the pending queue size."""
        return {
            "current_job_id": self.current_job_id,
            "queue_size": self.queue.qsize()
        }

    def _run_loop(self):
        """Worker thread loop to run training jobs sequentially."""
        while True:
            try:
                # Wait for a job with a timeout to allow the thread to stop if idle
                job_id = self.queue.get(timeout=10)
            except queue.Empty:
                break
                
            self.current_job_id = job_id
            db = SessionLocal()
            try:
                # 1. Update status to TRAINING
                job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
                if not job:
                    self.queue.task_done()
                    continue
                    
                dataset = db.query(DatasetUpload).filter(DatasetUpload.id == job.dataset_id).first()
                if not dataset:
                    job.status = "FAILED"
                    job.error_message = "Associated dataset upload records do not exist."
                    db.commit()
                    self.queue.task_done()
                    continue

                job.status = "TRAINING"
                job.started_at = datetime.utcnow()
                job.progress_percent = 0.0
                job.current_epoch = 0
                
                # Fetch data.yaml path from validation message
                # If validation succeeded, validation_message contains JSON or path
                # To be robust, find the yaml path recursively in the dataset root
                dataset_root = Path(dataset.storage_path) / "extracted"
                yaml_files = list(dataset_root.rglob("*.yaml")) + list(dataset_root.rglob("*.yml"))
                if not yaml_files:
                    raise FileNotFoundError("data.yaml file not found in extracted dataset folder.")
                
                yaml_path = yaml_files[0]
                
                # Define run folder
                run_dir = YOLO_RUNS_DIR / f"run_{job_id}"
                job.run_dir = str(run_dir.absolute())
                db.commit()

                # Fetch training parameters before closing database session
                epochs = job.epochs
                batch_size = job.batch_size
                imgsz = job.imgsz
                base_model = job.base_model

                # Close session before starting heavy training, so we don't hold lock
                db.close()

                # 2. Run YOLO Training
                self._execute_training(job_id, yaml_path, epochs, batch_size, imgsz, base_model, run_dir)
                
                # 3. Post-Training: Save weights and register Model Version
                db = SessionLocal()
                job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
                
                # Check if training succeeded and saved weights
                best_weights = run_dir / "weights" / "best.pt"
                if best_weights.exists():
                    # Copy weights to models registry directory
                    MODEL_DIR.mkdir(parents=True, exist_ok=True)
                    model_filename = f"model_v{job_id}.pt"
                    model_dest = MODEL_DIR / model_filename
                    shutil.copy(best_weights, model_dest)
                    
                    # Deactivate existing active models
                    db.query(ModelVersion).update({ModelVersion.is_active: False})
                    
                    # Create Model Version entry
                    version_name = f"YOLO_v{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    model_version = ModelVersion(
                        training_job_id=job_id,
                        version_name=version_name,
                        model_path=str(model_dest.absolute()),
                        is_active=True,
                        metrics=job.metrics
                    )
                    db.add(model_version)
                    
                    job.status = "COMPLETED"
                    job.progress_percent = 100.0
                else:
                    job.status = "FAILED"
                    job.error_message = "Training finished but no weights files (best.pt) were generated."
                
                job.completed_at = datetime.utcnow()
                db.commit()
                
            except Exception as e:
                # Catch training exceptions
                traceback.print_exc()
                db = SessionLocal()
                job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
                if job:
                    job.status = "FAILED"
                    job.completed_at = datetime.utcnow()
                    job.error_message = f"Training failed with error: {str(e)}"
                    db.commit()
            finally:
                db.close()
                self.queue.task_done()
                self.current_job_id = None

    def _execute_training(self, job_id: int, yaml_path: Path, epochs: int, batch: int, imgsz: int, base_model: str, run_dir: Path):
        """Handles the actual Ultralytics YOLO call with custom progress hooks, or simulated training in mock mode."""
        import time
        import random
        
        # Use default Nano model if not defined
        model_name = base_model if base_model else DEFAULT_BASE_MODEL
        
        if MOCK_TRAINING:
            print(f"Running in MOCK_TRAINING mode for Job {job_id}")
            
            # Download/ensure model file exists locally by instantiating it
            model = YOLO(model_name)
            
            # Start simulation loop
            total = epochs if epochs else 10
            for current in range(1, total + 1):
                time.sleep(0.5)  # Sleep 0.5s per epoch
                
                db = SessionLocal()
                try:
                    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
                    if job:
                        job.current_epoch = current
                        job.progress_percent = round((current / total) * 100, 1)
                        
                        # Generate dummy loss and MAP50 metrics
                        metrics_data = job.get_metrics() or {}
                        
                        # Simple model of training: loss decreases, mAP50 increases
                        dummy_loss = max(0.05, round(1.2 / (current ** 0.4) + random.uniform(-0.02, 0.02), 4))
                        dummy_map50 = min(0.99, round(0.4 + 0.55 * (1 - 1 / (current ** 0.5)) + random.uniform(-0.01, 0.01), 4))
                        dummy_precision = min(0.99, round(0.38 + 0.58 * (1 - 1 / (current ** 0.5)) + random.uniform(-0.01, 0.01), 4))
                        
                        epoch_metrics = {
                            "loss": dummy_loss,
                            "metrics_mAP50B": dummy_map50,
                            "metrics_precisionB": dummy_precision
                        }
                        
                        # Store history
                        history = metrics_data.get("history", [])
                        history.append({
                            "epoch": current,
                            **epoch_metrics
                        })
                        metrics_data["history"] = history
                        metrics_data["latest"] = epoch_metrics
                        
                        job.set_metrics(metrics_data)
                        db.commit()
                except Exception as e:
                    print(f"Error updating mock training progress: {e}")
                finally:
                    db.close()
                    
            # After loop finishes, create weights directory and copy model_name weights as best.pt
            weights_dir = run_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            
            src_pt = Path(model_name)
            if src_pt.exists():
                shutil.copy(src_pt, weights_dir / "best.pt")
            else:
                ckpt = getattr(model, 'ckpt_path', None)
                if ckpt and Path(ckpt).exists():
                    shutil.copy(ckpt, weights_dir / "best.pt")
                else:
                    with open(weights_dir / "best.pt", "wb") as f:
                        f.write(b"mock_weights")
            return

        # Load pre-trained model
        model = YOLO(model_name)
        
        # Define epoch callback function
        def on_train_epoch_end(trainer):
            db = SessionLocal()
            try:
                job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
                if job:
                    current = trainer.epoch + 1
                    total = trainer.epochs
                    job.current_epoch = current
                    job.progress_percent = round((current / total) * 100, 1)
                    
                    # Build metrics history
                    metrics_data = job.get_metrics() or {}
                    epoch_metrics = {}
                    
                    # Loss items
                    if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                        try:
                            if hasattr(trainer.loss_items, 'tolist'):
                                loss_list = trainer.loss_items.tolist()
                            else:
                                loss_list = list(trainer.loss_items)
                            epoch_metrics["loss"] = float(loss_list[0]) if loss_list else 0.0
                        except Exception:
                            pass
                    
                    # Try validation metrics from trainer.metrics
                    if hasattr(trainer, 'metrics') and trainer.metrics:
                        for k, v in trainer.metrics.items():
                            clean_key = k.replace("/", "_").replace("(", "").replace(")", "")
                            epoch_metrics[clean_key] = float(v)
                            
                    # Store history
                    history = metrics_data.get("history", [])
                    history.append({
                        "epoch": current,
                        **epoch_metrics
                    })
                    metrics_data["history"] = history
                    metrics_data["latest"] = epoch_metrics
                    
                    job.set_metrics(metrics_data)
                    db.commit()
            except Exception as e:
                print(f"Error updating training progress callback: {e}")
            finally:
                db.close()

        # Register callbacks
        model.add_callback("on_fit_epoch_end", on_train_epoch_end)
        
        # Auto-detect hardware: use CUDA GPU if available, else fall back to CPU
        import torch
        device_val = 0 if torch.cuda.is_available() else "cpu"
        print(f"Training Service: Starting YOLO on device={device_val}")

        # Start fine-tuning
        model.train(
            data=str(yaml_path.absolute().as_posix()),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            project=str(run_dir.parent.absolute().as_posix()),
            name=run_dir.name,
            device=device_val,
            plots=True,
            verbose=False
        )

# Global orchestrator instance
training_service = TrainingService()
