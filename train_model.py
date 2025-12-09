#!/usr/bin/env python3
"""
Complete training pipeline: Download datasets and train custom model.
"""

import os
import sys
from pathlib import Path
import yaml
from ultralytics import YOLO


def download_roboflow_dataset(api_key: str = None):
    """Download Roboflow dataset."""
    if not api_key:
        print("⚠️  No Roboflow API key provided. Skipping Roboflow dataset.")
        print("   Get your API key from: https://app.roboflow.com/")
        return None
    
    try:
        from roboflow import Roboflow
        
        print("📥 Downloading Roboflow dataset...")
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("boulderingdataset").project("climbing-holds-and-volumes-dohxi")
        dataset = project.version(1).download("yolov8", location="data/roboflow")
        
        print(f"✅ Roboflow dataset downloaded to: {dataset.location}")
        return dataset.location
    except Exception as e:
        print(f"❌ Error downloading Roboflow dataset: {e}")
        return None


def download_kaggle_dataset():
    """Download Kaggle dataset using kagglehub."""
    try:
        import kagglehub
        
        print("📥 Downloading Kaggle dataset...")
        path = kagglehub.dataset_download("tomasslama/indoor-climbing-gym-hold-segmentation")
        
        print(f"✅ Kaggle dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"❌ Error downloading Kaggle dataset: {e}")
        print("   Make sure you have: pip install kagglehub")
        return None


def prepare_dataset_config(dataset_path: str, output_path: str = "dataset_config.yaml"):
    """Create YOLOv8 dataset config."""
    dataset_path = Path(dataset_path).absolute()
    
    # Check if dataset has YOLOv8 structure
    if (dataset_path / "data.yaml").exists():
        print(f"✅ Found existing data.yaml in {dataset_path}")
        return str(dataset_path / "data.yaml")
    
    # Check if it's a Kaggle VIA format dataset (needs conversion)
    if any((dataset_path / f).is_dir() for f in ['bh', 'sm', 'bh-phone']):
        print("\n⚠️  Detected Kaggle dataset in VIA format.")
        print("   Converting to YOLOv8 format...")
        
        # Convert dataset
        from convert_via_to_yolo import convert_kaggle_dataset
        converted_path = convert_kaggle_dataset(str(dataset_path), "data/yolo_dataset")
        
        # Update dataset_path to converted location
        dataset_path = Path("data/yolo_dataset")
        output_path = "data/yolo_dataset/data.yaml"
        
        if Path(output_path).exists():
            print(f"✅ Using converted dataset: {output_path}")
            return output_path
        else:
            raise ValueError("Dataset conversion failed.")
    
    # Create config for YOLOv8 structure
    config = {
        'path': str(dataset_path),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 4,
        'names': {
            0: 'start_hold',
            1: 'intermediate_hold',
            2: 'finish_hold',
            3: 'volume'
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Dataset config created: {output_path}")
    return output_path


def train_model(
    data_config: str,
    model_size: str = 'n',
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 640
):
    """Train the hold detection model."""
    print(f"\n🚀 Starting model training...")
    print(f"   Model size: {model_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch}")
    print(f"   Image size: {imgsz}")
    
    # Load model
    model = YOLO(f'yolov8{model_size}-seg.pt')
    
    # Train
    # Use CPU on Mac (MPS has issues with NMS operation)
    import torch
    import os
    # Enable MPS fallback for operations not supported on MPS
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    device = 'cpu'  # Use CPU for stability on Mac
    
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project='runs/detect',
        name='hold_detector',
        patience=15,
        save=True,
        save_period=10,
        val=True,
        plots=True
    )
    
    # Find best model
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    
    if best_model_path.exists():
        # Copy to models directory
        import shutil
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        target_path = models_dir / "hold_detector.pt"
        shutil.copy(best_model_path, target_path)
        print(f"\n✅ Training complete!")
        print(f"   Best model saved to: {target_path}")
        print(f"   Training results: {results.save_dir}")
        return str(target_path)
    else:
        print(f"\n⚠️  Best model not found at expected location: {best_model_path}")
        return None


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("🧗 HoldOn - Model Training Pipeline")
    print("=" * 60)
    
    # Check for API keys
    roboflow_key = os.environ.get('ROBOFLOW_API_KEY')
    if not roboflow_key:
        print("\n💡 Tip: Set ROBOFLOW_API_KEY environment variable for Roboflow dataset")
        print("   export ROBOFLOW_API_KEY='your_key_here'")
    
    # Download datasets
    roboflow_path = None
    kaggle_path = None
    
    if roboflow_key:
        roboflow_path = download_roboflow_dataset(roboflow_key)
    
    kaggle_path = download_kaggle_dataset()
    
    # Determine which dataset to use
    # Prefer Roboflow as it's already in YOLOv8 format
    dataset_path = None
    if roboflow_path:
        dataset_path = roboflow_path
        print(f"\n✅ Using Roboflow dataset: {dataset_path}")
    elif kaggle_path:
        dataset_path = kaggle_path
        print(f"\n✅ Using Kaggle dataset: {dataset_path}")
        print("   (Will convert from VIA to YOLOv8 format if needed)")
    else:
        print("\n❌ No datasets available! Cannot train model.")
        print("   Please download at least one dataset:")
        print("   1. Set ROBOFLOW_API_KEY and run again (recommended)")
        print("   2. Or we can convert the Kaggle dataset (more complex)")
        return 1
    
    # Prepare dataset config
    data_config = prepare_dataset_config(dataset_path)
    
    # Train model
    model_path = train_model(
        data_config=data_config,
        model_size='n',  # nano for faster training
        epochs=1,        # Quick test with 1 epoch
        batch=16,
        imgsz=640
    )
    
    if model_path:
        print("\n" + "=" * 60)
        print("✅ Model training complete!")
        print(f"   Model saved to: {model_path}")
        print("   You can now use this model with the API server.")
        print("=" * 60)
        return 0
    else:
        print("\n❌ Training failed or model not found.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

