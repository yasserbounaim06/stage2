import time
import zipfile
import shutil
from pathlib import Path
import requests
from PIL import Image

def build_mock_dataset():
    print("Building mock dataset...")
    temp_dir = Path("./temp_mock_dataset")
    temp_dir.mkdir(exist_ok=True)
    
    # Create YOLO structure
    img_dir = temp_dir / "train" / "images"
    lbl_dir = temp_dir / "train" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    # Save a small white image
    img = Image.new("RGB", (100, 100), color="white")
    img.save(img_dir / "test_img.jpg")
    
    # Save a dummy label
    with open(lbl_dir / "test_img.txt", "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")
        
    # Save data.yaml
    yaml_content = "train: train/images\nval: train/images\nnames:\n  0: container_number\n"
    with open(temp_dir / "data.yaml", "w") as f:
        f.write(yaml_content)
        
    # Zip it up
    zip_path = Path("./temp_mock_dataset.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(temp_dir / "data.yaml", arcname="data.yaml")
        z.write(img_dir / "test_img.jpg", arcname="train/images/test_img.jpg")
        z.write(lbl_dir / "test_img.txt", arcname="train/labels/test_img.txt")
        
    # Clean up directory
    shutil.rmtree(temp_dir)
    print(f"Mock dataset zip created: {zip_path.absolute()}")
    return zip_path

def test_pipeline():
    base_url = "http://127.0.0.1:8000"
    zip_path = build_mock_dataset()
    
    try:
        # 1. Test Dataset Upload
        print("\n--- 1. Testing Dataset Upload ---")
        url = f"{base_url}/api/datasets/upload"
        with open(zip_path, "rb") as f:
            r = requests.post(url, data={"name": "Integration Test Dataset"}, files={"file": f})
        
        if r.status_code != 201:
            print(f"Upload failed: {r.status_code}")
            print(r.text)
            return
            
        dataset = r.json()
        print("Dataset uploaded successfully:")
        print(dataset)
        dataset_id = dataset["id"]
        
        # 2. Test Training Start
        print("\n--- 2. Testing Training Start ---")
        url = f"{base_url}/api/training/start"
        payload = {
            "dataset_id": dataset_id,
            "epochs": 5,
            "batch_size": 2,
            "imgsz": 640,
            "base_model": "yolov8n.pt"
        }
        r = requests.post(url, json=payload)
        if r.status_code != 202:
            print(f"Failed to start training: {r.status_code}")
            print(r.text)
            return
            
        job = r.json()
        print("Training job started successfully:")
        print(job)
        job_id = job["id"]
        
        # 3. Test Training Status Polling
        print("\n--- 3. Polling Training Status ---")
        url = f"{base_url}/api/training/status/{job_id}"
        
        while True:
            r = requests.get(url)
            if r.status_code != 200:
                print(f"Failed to fetch status: {r.status_code}")
                print(r.text)
                return
            
            job_status = r.json()
            status = job_status["status"]
            progress = job_status.get("progress_percent", 0.0)
            epoch = job_status.get("current_epoch", 0)
            metrics = job_status.get("metrics", {})
            
            print(f"Job ID: {job_id} | Status: {status} | Progress: {progress}% | Epoch: {epoch} | Metrics: {metrics}")
            
            if status in ["COMPLETED", "FAILED"]:
                break
                
            time.sleep(1)
            
        if status == "FAILED":
            print(f"Training job failed! Error: {job_status.get('error_message')}")
            return
            
        # 4. Test Inference
        print("\n--- 4. Testing Inference Pipeline ---")
        # Create a test query image
        query_img_path = Path("./temp_query_img.jpg")
        img = Image.new("RGB", (640, 640), color="blue")
        img.save(query_img_path)
        
        url = f"{base_url}/api/inference/detect"
        with open(query_img_path, "rb") as f:
            r = requests.post(url, files={"file": f})
            
        if r.status_code != 200:
            print(f"Inference failed: {r.status_code}")
            print(r.text)
            return
            
        result = r.json()
        print("Inference response:")
        print(result)
        
        print("\nPipeline integration test completed successfully!")
        
    except Exception as e:
        print(f"\nAn error occurred during verification: {e}")
    finally:
        # Cleanup
        if zip_path.exists():
            zip_path.unlink()
        if Path("./temp_query_img.jpg").exists():
            Path("./temp_query_img.jpg").unlink()

if __name__ == "__main__":
    test_pipeline()
