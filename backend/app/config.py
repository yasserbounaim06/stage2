import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

UPLOAD_DIR = DATA_DIR / "uploads"
MODEL_DIR = DATA_DIR / "models"
INFERENCE_DIR = DATA_DIR / "inference"
YOLO_RUNS_DIR = DATA_DIR / "yolo_runs"

# Ensure dirs exist
for path in [DATA_DIR, UPLOAD_DIR, MODEL_DIR, INFERENCE_DIR, YOLO_RUNS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/app.db")

# YOLO Defaults
DEFAULT_BASE_MODEL = "yolov8n.pt"
ALLOWED_EXTENSIONS = {".zip"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
LABEL_EXTENSIONS = {".txt"}

# Remote GPU Provider Keys
VAST_API_KEY = os.getenv("VAST_API_KEY", "")
SALAD_API_KEY = os.getenv("SALAD_API_KEY", "")
SALAD_ORG_NAME = os.getenv("SALAD_ORG_NAME", "")
SALAD_PROJECT_NAME = os.getenv("SALAD_PROJECT_NAME", "")

# The public address of this backend to let remote instances download datasets and upload weights.
PUBLIC_BACKEND_URL = os.getenv("PUBLIC_BACKEND_URL", "http://localhost:8000")


# Mock Training flag for environments without GPUs or resource constraints
MOCK_TRAINING = os.getenv("MOCK_TRAINING", "true").lower() == "true"
