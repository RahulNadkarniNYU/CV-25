"""
Script to download datasets from Roboflow and Kaggle.
"""

import os
from pathlib import Path
import argparse


def download_roboflow_dataset(api_key: str, output_dir: str = "data/roboflow"):
    """
    Download Roboflow dataset.
    Dataset: https://universe.roboflow.com/boulderingdataset/climbing-holds-and-volumes-dohxi
    """
    try:
        from roboflow import Roboflow
        
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("boulderingdataset").project("climbing-holds-and-volumes-dohxi")
        dataset = project.version(1).download("yolov8", location=output_dir)
        
        print(f"✅ Roboflow dataset downloaded to: {dataset.location}")
        return dataset.location
    except Exception as e:
        print(f"❌ Error downloading Roboflow dataset: {e}")
        print("Make sure you have installed: pip install roboflow")
        return None


def download_kaggle_dataset(dataset_name: str, output_dir: str = "data/kaggle"):
    """
    Download Kaggle dataset.
    Dataset: https://www.kaggle.com/datasets/tomasslama/indoor-climbing-gym-hold-segmentation
    """
    try:
        import kaggle
        
        # Download dataset
        kaggle.api.dataset_download_files(
            dataset_name,
            path=output_dir,
            unzip=True
        )
        
        print(f"✅ Kaggle dataset downloaded to: {output_dir}")
        return output_dir
    except Exception as e:
        print(f"❌ Error downloading Kaggle dataset: {e}")
        print("Make sure you have:")
        print("1. Installed: pip install kaggle")
        print("2. Set up Kaggle API credentials in ~/.kaggle/kaggle.json")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download bouldering datasets")
    parser.add_argument("--roboflow-key", type=str, help="Roboflow API key")
    parser.add_argument("--kaggle-dataset", type=str, 
                       default="tomasslama/indoor-climbing-gym-hold-segmentation",
                       help="Kaggle dataset name")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Download datasets
    if args.roboflow_key:
        download_roboflow_dataset(args.roboflow_key, 
                                  os.path.join(args.output_dir, "roboflow"))
    
    download_kaggle_dataset(args.kaggle_dataset,
                           os.path.join(args.output_dir, "kaggle"))

