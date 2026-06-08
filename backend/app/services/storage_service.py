import os
import shutil
import zipfile
from pathlib import Path
from app.config import UPLOAD_DIR, MODEL_DIR, INFERENCE_DIR

class StorageService:
    @staticmethod
    def save_upload(file_data: bytes, filename: str, dataset_id: int) -> Path:
        """Saves an uploaded zip file and returns the path."""
        dataset_dir = UPLOAD_DIR / str(dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        file_path = dataset_dir / filename
        with open(file_path, "wb") as f:
            f.write(file_data)
        return file_path

    @staticmethod
    def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
        """Extracts a zip file to the given directory."""
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        return extract_dir

    @staticmethod
    def get_dataset_dir(dataset_id: int) -> Path:
        return UPLOAD_DIR / str(dataset_id)

    @staticmethod
    def get_extracted_dir(dataset_id: int) -> Path:
        return UPLOAD_DIR / str(dataset_id) / "extracted"

    @staticmethod
    def clean_dataset_dir(dataset_id: int):
        dataset_dir = StorageService.get_dataset_dir(dataset_id)
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
            
    @staticmethod
    def save_inference_image(file_data: bytes, filename: str) -> tuple[Path, Path]:
        """Saves inference input image and returns (absolute_path, relative_web_path)."""
        INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        # Create a unique folder for each inference upload to avoid filename collisions
        import uuid
        run_id = str(uuid.uuid4())
        run_dir = INFERENCE_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = run_dir / filename
        with open(file_path, "wb") as f:
            f.write(file_data)
            
        relative_path = Path("inference") / run_id / filename
        return file_path, relative_path
