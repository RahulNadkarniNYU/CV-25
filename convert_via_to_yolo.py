#!/usr/bin/env python3
"""
Convert Kaggle VIA format dataset to YOLOv8 format.
"""

import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def convert_via_to_yolo(via_json_path: Path, images_dir: Path, output_dir: Path, split: str = 'train'):
    """Convert VIA annotations to YOLOv8 format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images_output = output_dir / 'images' / split
    labels_output = output_dir / 'labels' / split
    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)
    
    # Load VIA JSON
    with open(via_json_path, 'r') as f:
        via_data = json.load(f)
    
    # Process each image
    for key, file_data in tqdm(via_data.get('_via_img_metadata', {}).items(), desc=f"Converting {split}"):
        # Extract filename from metadata (key is like "0000.jpg4501555", filename is in file_data)
        filename = file_data.get('filename', key.split('.')[0] + '.jpg')
        
        if not filename.endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # Copy image
        src_image = images_dir / filename
        if not src_image.exists():
            print(f"⚠️  Image not found: {src_image}")
            continue
        
        dst_image = images_output / filename
        shutil.copy(src_image, dst_image)
        
        # Get image dimensions
        img = cv2.imread(str(src_image))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        
        # Create YOLO label file
        label_file = labels_output / (filename.rsplit('.', 1)[0] + '.txt')
        
        with open(label_file, 'w') as f:
            regions = file_data.get('regions', [])
            for region in regions:
                shape = region.get('shape_attributes', {})
                region_attrs = region.get('region_attributes', {})
                
                # Get class (default to intermediate_hold if not specified)
                class_name = region_attrs.get('class', 'intermediate_hold')
                
                # Map class names to IDs
                class_map = {
                    'start_hold': 0,
                    'intermediate_hold': 1,
                    'finish_hold': 2,
                    'volume': 3
                }
                class_id = class_map.get(class_name.lower(), 1)  # Default to intermediate
                
                # Convert polygon to YOLO format (normalized)
                if shape.get('name') == 'polygon':
                    all_points_x = shape.get('all_points_x', [])
                    all_points_y = shape.get('all_points_y', [])
                    
                    if len(all_points_x) >= 3:
                        # Convert to normalized polygon format for segmentation
                        points = []
                        for x, y in zip(all_points_x, all_points_y):
                            # Normalize coordinates
                            x_norm = x / img_w
                            y_norm = y / img_h
                            points.append(f"{x_norm:.6f} {y_norm:.6f}")
                        
                        # YOLOv8 segmentation format: class_id x1 y1 x2 y2 ...
                        f.write(f"{class_id} {' '.join(points)}\n")
                elif shape.get('name') == 'rect':
                    # Convert rectangle to polygon (4 corners)
                    x = shape.get('x', 0)
                    y = shape.get('y', 0)
                    w = shape.get('width', 0)
                    h = shape.get('height', 0)
                    
                    # Create 4 corners
                    corners = [
                        (x, y),
                        (x + w, y),
                        (x + w, y + h),
                        (x, y + h)
                    ]
                    
                    points = []
                    for cx, cy in corners:
                        x_norm = cx / img_w
                        y_norm = cy / img_h
                        points.append(f"{x_norm:.6f} {y_norm:.6f}")
                    
                    f.write(f"{class_id} {' '.join(points)}\n")


def convert_kaggle_dataset(kaggle_path: str, output_path: str = "data/yolo_dataset"):
    """Convert entire Kaggle dataset to YOLOv8 format."""
    kaggle_path = Path(kaggle_path)
    output_path = Path(output_path)
    
    print("🔄 Converting Kaggle VIA dataset to YOLOv8 format...")
    
    # Process each subfolder (bh, sm, bh-phone)
    splits = {
        'bh': 'train',  # Use bh as training
        'sm': 'valid',  # Use sm as validation
        'bh-phone': 'test'  # Use bh-phone as test
    }
    
    for folder, split in splits.items():
        folder_path = kaggle_path / folder
        json_path = kaggle_path / f"{folder}-annotation.json"
        
        if folder_path.exists() and json_path.exists():
            print(f"\n📁 Processing {folder} -> {split}...")
            convert_via_to_yolo(json_path, folder_path, output_path, split)
        else:
            print(f"⚠️  Skipping {folder} (not found)")
    
    # Create data.yaml
    config = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/valid',
        'test': 'images/test',
        'nc': 4,
        'names': {
            0: 'start_hold',
            1: 'intermediate_hold',
            2: 'finish_hold',
            3: 'volume'
        }
    }
    
    import yaml
    config_path = output_path / 'data.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n✅ Conversion complete!")
    print(f"   YOLOv8 dataset saved to: {output_path}")
    print(f"   Config file: {config_path}")
    return str(config_path)


if __name__ == "__main__":
    import sys
    
    kaggle_path = "~/.cache/kagglehub/datasets/tomasslama/indoor-climbing-gym-hold-segmentation/versions/4"
    if len(sys.argv) > 1:
        kaggle_path = sys.argv[1]
    
    kaggle_path = Path(kaggle_path).expanduser()
    
    if not kaggle_path.exists():
        print(f"❌ Dataset path not found: {kaggle_path}")
        sys.exit(1)
    
    convert_kaggle_dataset(kaggle_path)

