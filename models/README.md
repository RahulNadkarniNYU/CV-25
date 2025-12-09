# Models Directory

This directory stores trained models for hold detection.

## Model Files

- `hold_detector.pt`: Trained YOLOv8 segmentation model for hold detection
- `hold_detector.onnx`: Optimized ONNX model for mobile deployment (optional)

## Training Your Model

1. Download datasets using `scripts/download_datasets.py`
2. Train using `src/training/train_hold_detector.py`
3. Model will be saved to `runs/detect/hold_detector/weights/best.pt`
4. Copy to this directory: `cp runs/detect/hold_detector/weights/best.pt models/hold_detector.pt`

## Model Specifications

- **Architecture**: YOLOv8 Segmentation
- **Input Size**: 640x640 pixels
- **Classes**: 
  - 0: start_hold
  - 1: intermediate_hold
  - 2: finish_hold
  - 3: volume

## Using Pretrained Models

If you don't have a trained model yet, the system will use COCO pretrained weights for testing. For production use, train on climbing hold datasets.

