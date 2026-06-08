import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent))

# Standardize on 'app.' imports to match backend runtime namespace
from app.services.remote_training_service import remote_training_manager, VastAITrainer, SaladTrainer
from app.database import engine, Base, SessionLocal
from app.models import TrainingJob, DatasetUpload

class TestRemoteGPUIntegration(unittest.TestCase):
    def setUp(self):
        # Create database tables
        Base.metadata.create_all(bind=engine)
        self.db = SessionLocal()
        
        # Add a mock dataset record
        self.dataset = DatasetUpload(
            name="test_remote_ds",
            filename="mock.zip",
            storage_path="./mock_storage",
            is_validated=True
        )
        self.db.add(self.dataset)
        self.db.commit()
        self.db.refresh(self.dataset)

    def tearDown(self):
        self.db.delete(self.dataset)
        self.db.commit()
        self.db.close()

    @patch('requests.get')
    @patch('requests.put')
    def test_vast_ai_submission(self, mock_put, mock_get):
        # Mock Search bundles response
        mock_search_response = MagicMock()
        mock_search_response.status_code = 200
        mock_search_response.json.return_value = {
            "offers": [
                {"id": 4321, "gpu_name": "RTX 3060", "dph_total": 0.15}
            ]
        }
        mock_get.return_value = mock_search_response

        # Mock Rent allocation response
        mock_rent_response = MagicMock()
        mock_rent_response.status_code = 200
        mock_rent_response.json.return_value = {
            "success": True,
            "new_contract": 999888
        }
        mock_put.return_value = mock_rent_response

        # Temporary patch of the environment variables
        with patch('app.services.remote_training_service.VAST_API_KEY', 'fake_vast_key'):
            trainer = VastAITrainer()
            remote_id = trainer.submit_training_job(
                job_id=1,
                dataset_download_url="http://mock-ip/api/datasets/1/download",
                callback_url="http://mock-ip/api/training/complete/1",
                epochs=5,
                batch_size=8,
                imgsz=640,
                base_model="yolov8n.pt"
            )
            self.assertEqual(remote_id, "999888")
            
        print("Vast.ai API search and allocation mock test passed!")

    @patch('requests.post')
    def test_salad_submission(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "yolo-training-job-1"}
        mock_post.return_value = mock_response

        with patch('app.services.remote_training_service.SALAD_API_KEY', 'fake_salad_key'), \
             patch('app.services.remote_training_service.SALAD_ORG_NAME', 'my-org'), \
             patch('app.services.remote_training_service.SALAD_PROJECT_NAME', 'my-proj'):
            trainer = SaladTrainer()
            remote_id = trainer.submit_training_job(
                job_id=1,
                dataset_download_url="http://mock-ip/api/datasets/1/download",
                callback_url="http://mock-ip/api/training/complete/1",
                epochs=5,
                batch_size=8,
                imgsz=640,
                base_model="yolov8n.pt"
            )
            self.assertEqual(remote_id, "yolo-training-job-1")
            
        print("Salad.cloud API deployment mock test passed!")

if __name__ == "__main__":
    unittest.main()
