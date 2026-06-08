import os
import yaml
from pathlib import Path
from app.config import IMAGE_EXTENSIONS, LABEL_EXTENSIONS

class DatasetService:
    @staticmethod
    def validate_dataset(extracted_path: Path) -> dict:
        """
        Validates the extracted YOLO dataset structure.
        Returns a dict with validation status, metrics, and message.
        """
        # 1. Find data.yaml recursively
        yaml_files = list(extracted_path.rglob("*.yaml")) + list(extracted_path.rglob("*.yml"))
        if not yaml_files:
            return {
                "valid": False,
                "message": "No data.yaml or data.yml found in the dataset upload.",
                "num_images": 0,
                "num_labels": 0
            }
        
        yaml_path = yaml_files[0]
        dataset_root = yaml_path.parent
        
        # 2. Parse data.yaml
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data_config = yaml.safe_load(f)
        except Exception as e:
            return {
                "valid": False,
                "message": f"Failed to parse data.yaml: {str(e)}",
                "num_images": 0,
                "num_labels": 0
            }
        
        if not isinstance(data_config, dict):
            return {
                "valid": False,
                "message": "data.yaml is invalid (should be a YAML mapping).",
                "num_images": 0,
                "num_labels": 0
            }
            
        # Check required fields
        if "names" not in data_config:
            return {
                "valid": False,
                "message": "data.yaml is missing the 'names' key (class names dictionary or list).",
                "num_images": 0,
                "num_labels": 0
            }
            
        # Determine paths
        train_path = data_config.get("train")
        val_path = data_config.get("val")
        
        if not train_path:
            return {
                "valid": False,
                "message": "data.yaml is missing the 'train' folder configuration.",
                "num_images": 0,
                "num_labels": 0
            }
            
        # 3. Locate files
        # Check if train folder is absolute or relative
        train_images_dir = dataset_root / train_path if not os.path.isabs(train_path) else Path(train_path)
        
        # If path field exists in yaml, it might change base folder
        yaml_path_field = data_config.get("path")
        if yaml_path_field:
            base_path = Path(yaml_path_field)
            if not base_path.is_absolute():
                base_path = dataset_root / base_path
            train_images_dir = base_path / train_path

        # If not found directly, try fallback: look for standard "train/images" folder in dataset_root
        if not train_images_dir.exists():
            train_images_dir = dataset_root / "train" / "images"
            if not train_images_dir.exists():
                train_images_dir = dataset_root / "images" # check if flat structure
                
        if not train_images_dir.exists():
            return {
                "valid": False,
                "message": f"Could not find training images directory (searched '{train_path}').",
                "num_images": 0,
                "num_labels": 0
            }

        # Locate training labels directory
        # Standard YOLO structure mirrors labels/ and images/
        train_labels_dir = Path(str(train_images_dir).replace("images", "labels"))
        if not train_labels_dir.exists() and train_images_dir.parent != dataset_root:
            train_labels_dir = train_images_dir.parent / "labels"
            
        if not train_labels_dir.exists():
            return {
                "valid": False,
                "message": f"Could not find matching training labels directory (searched '{train_labels_dir}').",
                "num_images": 0,
                "num_labels": 0
            }

        # 4. Count and cross-verify files
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(list(train_images_dir.rglob(f"*{ext}")))
            image_files.extend(list(train_images_dir.rglob(f"*{ext.upper()}")))
            
        label_files = list(train_labels_dir.rglob("*.txt"))
        
        if not image_files:
            return {
                "valid": False,
                "message": "No images found in training directory.",
                "num_images": 0,
                "num_labels": 0
            }
            
        # Validate format of labels and coordinates
        invalid_labels_count = 0
        error_sample = ""
        
        for lf in label_files[:50]: # check up to 50 labels for formatting
            try:
                with open(lf, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        invalid_labels_count += 1
                        error_sample = f"Label line does not have at least 5 columns: '{line}' in file {lf.name}"
                        break
                    # Class index check
                    class_idx = int(parts[0])
                    # Coordinates check (must be floats between 0 and 1, with a tiny tolerance for rounding)
                    coords = [float(x) for x in parts[1:]]
                    for coord in coords:
                        if coord < -0.05 or coord > 1.05:
                            invalid_labels_count += 1
                            error_sample = f"Coordinate out of range [-0.05, 1.05]: {coord} in file {lf.name}"
                            break
                    if invalid_labels_count > 0:
                        break
            except Exception as e:
                invalid_labels_count += 1
                error_sample = f"Failed to read label file {lf.name}: {str(e)}"
                break
                
        if invalid_labels_count > 0:
            return {
                "valid": False,
                "message": f"Label format error: {error_sample}",
                "num_images": len(image_files),
                "num_labels": len(label_files)
            }

        # 5. Rewrite/Reconcile data.yaml to make it absolute
        # This guarantees that the Ultralytics training engine running in any context will locate the files correctly
        try:
            # Check for val directory fallback
            val_images_dir = dataset_root / val_path if val_path and not os.path.isabs(val_path) else (Path(val_path) if val_path else None)
            if yaml_path_field and val_path:
                val_images_dir = Path(yaml_path_field) / val_path
                if not val_images_dir.is_absolute():
                    val_images_dir = dataset_root / val_images_dir

            if not val_images_dir or not val_images_dir.exists():
                val_images_dir = dataset_root / "val" / "images"
                if not val_images_dir.exists():
                    val_images_dir = train_images_dir  # fallback validation to training if not present
            
            # Formulate new absolute YAML paths
            reconciled_config = {
                "path": str(dataset_root.absolute().as_posix()),
                "train": str(train_images_dir.absolute().relative_to(dataset_root.absolute()).as_posix()),
                "val": str(val_images_dir.absolute().relative_to(dataset_root.absolute()).as_posix()),
                "names": data_config["names"]
            }
            
            # Save the reconciled config back to the data.yaml file
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(reconciled_config, f, default_flow_style=False)
                
        except Exception as e:
            return {
                "valid": False,
                "message": f"Failed to reconcile data.yaml paths: {str(e)}",
                "num_images": len(image_files),
                "num_labels": len(label_files)
            }

        return {
            "valid": True,
            "message": f"Dataset is valid. Found {len(image_files)} training images and {len(label_files)} labels.",
            "num_images": len(image_files),
            "num_labels": len(label_files),
            "yaml_path": str(yaml_path.absolute()),
            "dataset_root": str(dataset_root.absolute())
        }
