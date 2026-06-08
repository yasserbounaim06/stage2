import os
import sys
import json
import urllib.parse
import requests
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session

# Path helper to support running from root or backend folder
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.config import VAST_API_KEY, SALAD_API_KEY, SALAD_ORG_NAME, SALAD_PROJECT_NAME, PUBLIC_BACKEND_URL
from app.database import SessionLocal
from app.models import TrainingJob, ModelVersion

# Base Provider Interface
class BaseTrainer(ABC):
    @abstractmethod
    def submit_training_job(
        self, 
        job_id: int, 
        dataset_download_url: str, 
        callback_url: str, 
        epochs: int, 
        batch_size: int, 
        imgsz: int, 
        base_model: str
    ) -> str:
        pass

    @abstractmethod
    def get_training_status(self, remote_job_id: str) -> str:
        pass

    @abstractmethod
    def get_training_logs(self, remote_job_id: str) -> str:
        pass

    @abstractmethod
    def cancel_training_job(self, remote_job_id: str) -> bool:
        pass


class VastAITrainer(BaseTrainer):
    def _get_headers(self):
        return {"Authorization": f"Bearer {VAST_API_KEY}"} if VAST_API_KEY else {}

    def submit_training_job(self, job_id: int, dataset_download_url: str, callback_url: str, epochs: int, batch_size: int, imgsz: int, base_model: str) -> str:
        if not VAST_API_KEY:
            raise ValueError("Vast.ai API key is not configured in environment variables.")

        query = {
            "rentable": True,
            "verified": True,
            "allocated": False,
            "num_gpus": 1,
            "inet_down": {"gt": 40.0}
        }
        
        query_str = urllib.parse.quote(json.dumps(query))
        search_url = f"https://console.vast.ai/api/v0/bundles/?q={query_str}&api_key={VAST_API_KEY}"
        
        print(f"Vast.ai: Searching offers using url {search_url}...")
        r = requests.get(search_url, headers=self._get_headers())
        if r.status_code != 200:
            raise RuntimeError(f"Vast.ai search failed: {r.text}")
            
        data = r.json()
        offers = data.get("offers", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        
        if not offers:
            raise RuntimeError("Vast.ai: No available GPU offers match the search filters.")
            
        offers = sorted(offers, key=lambda x: x.get("dph_total", 999.0))
        cheapest_offer = offers[0]
        offer_id = cheapest_offer.get("id")
        gpu_name = cheapest_offer.get("gpu_name", "GPU")
        price = cheapest_offer.get("dph_total", 0.0)
        print(f"Vast.ai: Selected offer {offer_id} ({gpu_name}) costing ${price:.3f}/hr.")

        # Script to run inside container
        worker_script = f"""import urllib.request, zipfile, os, requests, csv, json

dataset_url = "{dataset_download_url}"
callback_url = "{callback_url}"
epochs = {epochs}
batch = {batch_size}
imgsz = {imgsz}
base_model = "{base_model}"

print("YOLO REMOTE WORKER STARTING...")
try:
    print("Downloading dataset ZIP from: " + dataset_url)
    urllib.request.urlretrieve(dataset_url, "dataset.zip")
    
    print("Extracting dataset...")
    with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
        zip_ref.extractall("dataset")
        
    yaml_files = []
    for root, dirs, files in os.walk("dataset"):
        for file in files:
            if file in ["data.yaml", "data.yml"]:
                yaml_files.append(os.path.join(root, file))
                
    if not yaml_files:
        raise FileNotFoundError("data.yaml config not found in extracted dataset ZIP")
        
    yaml_path = yaml_files[0]
    print("Found dataset YAML at: " + yaml_path)
    
    print("Starting YOLO train command on GPU...")
    cmd = f"yolo train data='{{yaml_path}}' epochs={{epochs}} batch={{batch}} imgsz={{imgsz}} model='{{base_model}}' project='runs' name='train' device=0 plots=False"
    ret = os.system(cmd)
    
    metrics_data = {{"history": [], "latest": {{}}}}
    csv_path = "runs/train/results.csv"
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean_row = {{}}
                for k, v in row.items():
                    clean_k = k.strip().replace("/", "_").replace("(", "").replace(")", "")
                    try:
                        clean_row[clean_k] = float(v.strip())
                    except ValueError:
                        clean_row[clean_k] = v.strip()
                metrics_data["history"].append(clean_row)
            if metrics_data["history"]:
                metrics_data["latest"] = metrics_data["history"][-1]
                
    best_weights = "runs/train/weights/best.pt"
    if os.path.exists(best_weights) and ret == 0:
        print("Training succeeded. Uploading weights to backend...")
        with open(best_weights, "rb") as f:
            r = requests.post(
                callback_url, 
                files={{"file": f}}, 
                data={{"status": "completed", "metrics": json.dumps(metrics_data)}}
            )
            print("Callback sent. Response: " + str(r.status_code))
    else:
        raise RuntimeError("YOLO training process failed or did not save best.pt.")
        
except Exception as e:
    err_str = str(e)
    print("Execution Error: " + err_str)
    requests.post(callback_url, data={{"status": "failed", "error": err_str}})
"""
        onstart_cmd = f"cat << 'EOF' > worker.py\n{worker_script}\nEOF\npython3 worker.py"

        rent_url = f"https://console.vast.ai/api/v0/asks/{offer_id}/?api_key={VAST_API_KEY}"
        payload = {
            "image": "ultralytics/ultralytics:latest",
            "disk": 20.0,
            "runtype": "ssh",
            "onstart": onstart_cmd
        }
        
        r = requests.put(rent_url, json=payload, headers=self._get_headers())
        if r.status_code != 200:
            raise RuntimeError(f"Vast.ai rent contract allocation failed: {r.text}")
            
        res_data = r.json()
        instance_id = res_data.get("new_contract") or res_data.get("id")
        if not instance_id:
            raise RuntimeError(f"Vast.ai did not return a valid instance ID: {res_data}")
            
        print(f"Vast.ai: Successfully allocated instance {instance_id}")
        return str(instance_id)

    def get_training_status(self, remote_job_id: str) -> str:
        url = f"https://console.vast.ai/api/v0/instances/{remote_job_id}/?api_key={VAST_API_KEY}"
        r = requests.get(url, headers=self._get_headers())
        if r.status_code != 200:
            print(f"Vast.ai: Failed to check status of {remote_job_id}: {r.text}")
            return "FAILED"
            
        data = r.json()
        instance = data.get("instance", {}) if isinstance(data, dict) else {}
        actual_status = instance.get("actual_status", "").lower()
        cur_state = instance.get("cur_state", "").lower()
        
        print(f"Vast.ai: Instance {remote_job_id} status: '{actual_status}', state: '{cur_state}'")
        
        if actual_status in ["loading", "starting", "initializing"]:
            return "PENDING"
        elif actual_status == "running" or cur_state == "running":
            return "TRAINING"
        elif actual_status == "stopped":
            return "FAILED"
        elif actual_status == "offline" or cur_state == "offline":
            return "FAILED"
        return "TRAINING"

    def get_training_logs(self, remote_job_id: str) -> str:
        url = f"https://console.vast.ai/api/v0/instances/{remote_job_id}/logs/?api_key={VAST_API_KEY}"
        r = requests.get(url, headers=self._get_headers())
        if r.status_code == 200:
            return r.json().get("stdout", "") or r.json().get("stderr", "") or "No logs available."
        return f"Failed to retrieve logs from Vast.ai: {r.text}"

    def cancel_training_job(self, remote_job_id: str) -> bool:
        url = f"https://console.vast.ai/api/v0/instances/{remote_job_id}/?api_key={VAST_API_KEY}"
        r = requests.delete(url, headers=self._get_headers())
        if r.status_code == 200:
            print(f"Vast.ai: Destroyed instance {remote_job_id} successfully.")
            return True
        print(f"Vast.ai: Failed to destroy instance {remote_job_id}: {r.text}")
        return False


class SaladTrainer(BaseTrainer):
    def _get_headers(self):
        return {
            "Salad-Api-Key": SALAD_API_KEY,
            "Content-Type": "application/json"
        }

    def submit_training_job(self, job_id: int, dataset_download_url: str, callback_url: str, epochs: int, batch_size: int, imgsz: int, base_model: str) -> str:
        if not SALAD_API_KEY or not SALAD_ORG_NAME or not SALAD_PROJECT_NAME:
            raise ValueError("Salad API keys or organization/project names are missing.")

        url = f"https://api.salad.com/api/v1/organizations/{SALAD_ORG_NAME}/projects/{SALAD_PROJECT_NAME}/containers"
        container_group_name = f"yolo-training-job-{job_id}"
        
        worker_script = f"""import urllib.request, zipfile, os, requests, csv, json

dataset_url = "{dataset_download_url}"
callback_url = "{callback_url}"
epochs = {epochs}
batch = {batch_size}
imgsz = {imgsz}
base_model = "{base_model}"

print("SALAD WORKER STARTING...")
try:
    print("Downloading dataset ZIP...")
    urllib.request.urlretrieve(dataset_url, "dataset.zip")
    print("Extracting...")
    with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
        zip_ref.extractall("dataset")
        
    yaml_files = []
    for root, dirs, files in os.walk("dataset"):
        for file in files:
            if file in ["data.yaml", "data.yml"]:
                yaml_files.append(os.path.join(root, file))
                
    if not yaml_files:
        raise FileNotFoundError("data.yaml config not found in dataset ZIP")
        
    yaml_path = yaml_files[0]
    
    print("Running YOLO training...")
    cmd = f"yolo train data='{{yaml_path}}' epochs={{epochs}} batch={{batch}} imgsz={{imgsz}} model='{{base_model}}' project='runs' name='train' device=0 plots=False"
    ret = os.system(cmd)
    
    metrics_data = {{"history": [], "latest": {{}}}}
    csv_path = "runs/train/results.csv"
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean_row = {{}}
                for k, v in row.items():
                    clean_k = k.strip().replace("/", "_").replace("(", "").replace(")", "")
                    try:
                        clean_row[clean_k] = float(v.strip())
                    except ValueError:
                        clean_row[clean_k] = v.strip()
                metrics_data["history"].append(clean_row)
            if metrics_data["history"]:
                metrics_data["latest"] = metrics_data["history"][-1]
                
    best_weights = "runs/train/weights/best.pt"
    if os.path.exists(best_weights) and ret == 0:
        print("Training succeeded. Uploading weights...")
        with open(best_weights, "rb") as f:
            requests.post(callback_url, files={{"file": f}}, data={{"status": "completed", "metrics": json.dumps(metrics_data)}})
    else:
        raise RuntimeError("YOLO training failed or best.pt not generated.")
        
except Exception as e:
    requests.post(callback_url, data={{"status": "failed", "error": str(e)}})
"""
        onstart_cmd = f"cat << 'EOF' > worker.py\n{worker_script}\nEOF\npython3 worker.py"
        
        payload = {
            "name": container_group_name,
            "container": {
                "image": "ultralytics/ultralytics:latest",
                "resources": {
                    "cpu": 2,
                    "memory": 8192,
                    "gpu_classes": ["rtx-3060", "rtx-4060"]
                },
                "command": ["bash", "-c", onstart_cmd]
            },
            "replica_count": 1,
            "restart_policy": "never"
        }
        
        r = requests.post(url, json=payload, headers=self._get_headers())
        if r.status_code != 201:
            raise RuntimeError(f"Salad container group creation failed: {r.text}")
            
        print(f"Salad: Successfully created container group {container_group_name}")
        return container_group_name

    def get_training_status(self, remote_job_id: str) -> str:
        url = f"https://api.salad.com/api/v1/organizations/{SALAD_ORG_NAME}/projects/{SALAD_PROJECT_NAME}/containers/{remote_job_id}"
        r = requests.get(url, headers=self._get_headers())
        if r.status_code != 200:
            print(f"Salad: Failed to fetch status for group {remote_job_id}: {r.text}")
            return "FAILED"
            
        data = r.json()
        status = data.get("status", "").lower()
        
        print(f"Salad: Container group {remote_job_id} status: '{status}'")
        
        if status in ["pending", "preparing", "deploying"]:
            return "PENDING"
        elif status == "running":
            return "TRAINING"
        elif status == "stopped":
            return "COMPLETED"
        elif status == "failed":
            return "FAILED"
        return "TRAINING"

    def get_training_logs(self, remote_job_id: str) -> str:
        url = f"https://api.salad.com/api/v1/organizations/{SALAD_ORG_NAME}/projects/{SALAD_PROJECT_NAME}/containers/{remote_job_id}/instances"
        r = requests.get(url, headers=self._get_headers())
        if r.status_code == 200:
            instances = r.json().get("instances", [])
            if instances:
                instance_id = instances[0].get("id")
                logs_url = f"https://api.salad.com/api/v1/organizations/{SALAD_ORG_NAME}/projects/{SALAD_PROJECT_NAME}/containers/{remote_job_id}/instances/{instance_id}/logs"
                log_r = requests.get(logs_url, headers=self._get_headers())
                if log_r.status_code == 200:
                    return log_r.json().get("stdout", "") or log_r.json().get("stderr", "") or "No stdout logs."
        return "Failed to fetch logs from Salad.cloud"

    def cancel_training_job(self, remote_job_id: str) -> bool:
        url = f"https://api.salad.com/api/v1/organizations/{SALAD_ORG_NAME}/projects/{SALAD_PROJECT_NAME}/containers/{remote_job_id}"
        r = requests.delete(url, headers=self._get_headers())
        if r.status_code in [200, 204]:
            print(f"Salad: Terminated and deleted container group {remote_job_id}.")
            return True
        print(f"Salad: Failed to delete container group {remote_job_id}: {r.text}")
        return False


# Global Orchestrator Manager
class RemoteTrainingManager:
    def __init__(self):
        self.vast_trainer = VastAITrainer()
        self.salad_trainer = SaladTrainer()

    def start_training(self, job_id: int, epochs: int, batch_size: int, imgsz: int, base_model: str):
        dataset_download_url = f"{PUBLIC_BACKEND_URL}/api/datasets/{job_id}/download"
        callback_url = f"{PUBLIC_BACKEND_URL}/api/training/complete/{job_id}"

        db = SessionLocal()
        try:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if not job:
                return

            provider_used = None
            remote_id = None
            error_log = []

            # 1. Try Vast.ai
            if VAST_API_KEY:
                try:
                    remote_id = self.vast_trainer.submit_training_job(
                        job_id, dataset_download_url, callback_url, epochs, batch_size, imgsz, base_model
                    )
                    provider_used = "vastai"
                except Exception as e:
                    traceback.print_exc()
                    error_log.append(f"Vast.ai submit failed: {str(e)}")
            else:
                error_log.append("Vast.ai skipped (VAST_API_KEY not configured).")

            # 2. Try Salad.cloud Fallback
            if not provider_used and SALAD_API_KEY:
                try:
                    remote_id = self.salad_trainer.submit_training_job(
                        job_id, dataset_download_url, callback_url, epochs, batch_size, imgsz, base_model
                    )
                    provider_used = "salad"
                except Exception as e:
                    traceback.print_exc()
                    error_log.append(f"Salad submit failed: {str(e)}")
            elif not provider_used:
                error_log.append("Salad.cloud skipped (SALAD_API_KEY not configured).")

            # 3. Handle allocation outcomes
            if provider_used and remote_id:
                job.provider = provider_used
                job.remote_job_id = remote_id
                job.remote_status = "PENDING"
                job.status = "PENDING"
                job.started_at = datetime.utcnow()
                db.commit()
                print(f"RemoteTrainingManager: Job {job_id} submitted to {provider_used} with ID {remote_id}.")
            else:
                failed_reason = " | ".join(error_log)
                job.status = "FAILED"
                job.completed_at = datetime.utcnow()
                job.error_message = f"Could not allocate remote GPU instance: {failed_reason}"
                db.commit()
                print(f"RemoteTrainingManager: Job {job_id} failed allocation: {failed_reason}")
                
        except Exception as e:
            traceback.print_exc()
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if job:
                job.status = "FAILED"
                job.completed_at = datetime.utcnow()
                job.error_message = f"Submission logic error: {str(e)}"
                db.commit()
        finally:
            db.close()

    def sync_job_status(self, job_id: int) -> tuple[str, str]:
        db = SessionLocal()
        try:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if not job or job.status not in ["PENDING", "TRAINING"] or not job.remote_job_id:
                return (job.status if job else "FAILED", job.remote_status if job else "FAILED")

            status_update = None
            if job.provider == "vastai":
                status_update = self.vast_trainer.get_training_status(job.remote_job_id)
            elif job.provider == "salad":
                status_update = self.salad_trainer.get_training_status(job.remote_job_id)

            if status_update:
                job.remote_status = status_update
                if status_update == "PENDING":
                    job.status = "PENDING"
                elif status_update == "TRAINING":
                    job.status = "TRAINING"
                elif status_update == "FAILED":
                    job.status = "FAILED"
                    job.completed_at = datetime.utcnow()
                    job.error_message = "Remote GPU instance reported error or shut down prematurely."
                    self.cancel_training_job(job_id)
                db.commit()

            return job.status, job.remote_status
        except Exception as e:
            print(f"RemoteTrainingManager: Failed to sync status for job {job_id}: {e}")
            return "FAILED", "FAILED"
        finally:
            db.close()

    def cancel_training_job(self, job_id: int) -> bool:
        db = SessionLocal()
        try:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if not job or not job.remote_job_id:
                return False

            success = False
            if job.provider == "vastai":
                success = self.vast_trainer.cancel_training_job(job.remote_job_id)
            elif job.provider == "salad":
                success = self.salad_trainer.cancel_training_job(job.remote_job_id)

            if success:
                job.remote_status = "CANCELLED"
                job.status = "FAILED"
                job.error_message = "Training run cancelled by user."
                job.completed_at = datetime.utcnow()
                db.commit()
            return success
        except Exception as e:
            print(f"RemoteTrainingManager: Cancel job {job_id} error: {e}")
            return False
        finally:
            db.close()

    def fetch_logs(self, job_id: int) -> str:
        db = SessionLocal()
        try:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if not job or not job.remote_job_id:
                return "No remote logs available."

            logs = ""
            if job.provider == "vastai":
                logs = self.vast_trainer.get_training_logs(job.remote_job_id)
            elif job.provider == "salad":
                logs = self.salad_trainer.get_training_logs(job.remote_job_id)

            if logs:
                job.training_logs = logs
                db.commit()
            return logs or "Logs are empty."
        except Exception as e:
            return f"Failed to fetch logs: {str(e)}"
        finally:
            db.close()


remote_training_manager = RemoteTrainingManager()
