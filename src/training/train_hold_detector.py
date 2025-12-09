"""
Training Script for Hold Detection Model
Trains YOLOv8 model on climbing hold datasets.
"""

import os
from pathlib import Path
from ultralytics import YOLO
import yaml


def train_hold_detector(
    data_config: str,
    model_size: str = 'n',  # n, s, m, l, x
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = 'auto',
    project: str = 'runs/detect',
    name: str = 'hold_detector'
):
    """
    Train YOLOv8 model for hold detection.
    
    Args:
        data_config: Path to dataset YAML config file
        model_size: Model size (nano, small, medium, large, xlarge)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        device: Device to train on ('cpu', 'cuda', or 'auto')
        project: Project directory for outputs
        name: Experiment name
    """
    # Load model
    model = YOLO(f'yolov8{model_size}-seg.pt')  # Segmentation model
    
    # Train the model
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=20,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        val=True,
        plots=True
    )
    
    print(f"Training completed! Best model saved at: {results.save_dir}")
    return results


def download_roboflow_dataset(api_key: str, workspace: str, project: str, version: int):
    """
    Download dataset from Roboflow.
    
    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project: Project name
        version: Dataset version number
    """
    from roboflow import Roboflow
    
    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    dataset = project_obj.version(version).download("yolov8")
    
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset.location


def create_dataset_config(dataset_path: str, output_path: str):
    """
    Create YAML config file for YOLOv8 training.
    
    Args:
        dataset_path: Path to dataset directory
        output_path: Path to save config YAML
    """
    config = {
        'path': str(Path(dataset_path).absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 4,  # Number of classes
        'names': {
            0: 'start_hold',
            1: 'intermediate_hold',
            2: 'finish_hold',
            3: 'volume'
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Dataset config saved to: {output_path}")
    return output_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train hold detection model')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset YAML config')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    
    args = parser.parse_args()
    
    train_hold_detector(
        data_config=args.data,
        model_size=args.model_size,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz
    )

