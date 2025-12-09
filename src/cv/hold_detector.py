"""
Hold Detection Module
Detects and classifies climbing holds in bouldering wall images.
Uses YOLOv8 for object detection and instance segmentation.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from pathlib import Path
import torch


class HoldDetector:
    """
    Detects and classifies climbing holds using YOLOv8.
    Supports detection of start holds, intermediate holds, and finish holds.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize the hold detector.
        
        Args:
            model_path: Path to trained YOLOv8 model. If None, uses pretrained weights.
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
        """
        self.device = self._get_device(device)
        
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # Initialize with pretrained COCO weights, will need fine-tuning
            self.model = YOLO('yolov8n-seg.pt')  # Segmentation model for precise hold boundaries
        
        self.model.to(self.device)
        
        # Hold class mappings (will be updated based on dataset)
        self.hold_classes = {
            0: 'start_hold',
            1: 'intermediate_hold',
            2: 'finish_hold',
            3: 'volume'  # Climbing volumes/wall features
        }
    
    def _get_device(self, device: str) -> str:
        """Determine the best available device."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def detect_holds(self, image: np.ndarray, conf_threshold: float = 0.25) -> Dict:
        """
        Detect all holds in the image.
        
        Args:
            image: Input image as numpy array (BGR format)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Dictionary containing:
                - holds: List of detected holds with bounding boxes, classes, and masks
                - image_shape: Original image dimensions
        """
        # Run inference
        results = self.model(image, conf=conf_threshold, verbose=False)
        
        holds = []
        
        if len(results) > 0 and results[0].masks is not None:
            # Get masks, boxes, and classes
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for i, (mask, box, cls, conf) in enumerate(zip(masks, boxes, classes, confidences)):
                # Get hold center and bounding box
                y_coords, x_coords = np.where(mask > 0.5)
                
                if len(y_coords) > 0 and len(x_coords) > 0:
                    center_y = int(np.mean(y_coords))
                    center_x = int(np.mean(x_coords))
                    
                    hold_info = {
                        'id': i,
                        'class': self.hold_classes.get(cls, 'unknown'),
                        'class_id': int(cls),
                        'confidence': float(conf),
                        'bbox': {
                            'x1': float(box[0]),
                            'y1': float(box[1]),
                            'x2': float(box[2]),
                            'y2': float(box[3])
                        },
                        'center': {'x': center_x, 'y': center_y},
                        'mask': mask,
                        'area': int(np.sum(mask > 0.5))
                    }
                    holds.append(hold_info)
        
        return {
            'holds': holds,
            'image_shape': image.shape[:2],
            'num_holds': len(holds)
        }
    
    def visualize_detections(self, image: np.ndarray, detections: Dict, 
                           show_labels: bool = True) -> np.ndarray:
        """
        Visualize detected holds on the image.
        
        Args:
            image: Original image
            detections: Detection results from detect_holds()
            show_labels: Whether to show class labels
            
        Returns:
            Image with visualizations overlaid
        """
        vis_image = image.copy()
        
        # Color mapping for different hold types
        colors = {
            'start_hold': (0, 255, 0),      # Green
            'intermediate_hold': (255, 165, 0),  # Orange
            'finish_hold': (255, 0, 0),     # Red
            'volume': (128, 128, 128),      # Gray
            'unknown': (255, 255, 255)      # White
        }
        
        for hold in detections['holds']:
            bbox = hold['bbox']
            hold_class = hold['class']
            conf = hold['confidence']
            center = hold['center']
            
            color = colors.get(hold_class, colors['unknown'])
            
            # Draw bounding box
            cv2.rectangle(vis_image,
                         (int(bbox['x1']), int(bbox['y1'])),
                         (int(bbox['x2']), int(bbox['y2'])),
                         color, 2)
            
            # Draw center point
            cv2.circle(vis_image, (center['x'], center['y']), 5, color, -1)
            
            # Draw mask overlay (semi-transparent)
            if 'mask' in hold:
                mask = hold['mask']
                mask_resized = cv2.resize(mask.astype(np.uint8) * 255,
                                        (image.shape[1], image.shape[0]))
                overlay = vis_image.copy()
                overlay[mask_resized > 127] = color
                vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
            
            # Draw label
            if show_labels:
                label = f"{hold_class} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(vis_image,
                            (int(bbox['x1']), int(bbox['y1']) - label_size[1] - 10),
                            (int(bbox['x1']) + label_size[0], int(bbox['y1'])),
                            color, -1)
                cv2.putText(vis_image, label,
                          (int(bbox['x1']), int(bbox['y1']) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return vis_image
    
    def filter_holds_by_type(self, detections: Dict, hold_type: str) -> List[Dict]:
        """Filter holds by type (start_hold, intermediate_hold, finish_hold, volume)."""
        return [h for h in detections['holds'] if h['class'] == hold_type]
    
    def get_route_structure(self, detections: Dict) -> Dict:
        """
        Analyze the route structure from detections.
        
        Returns:
            Dictionary with route information including start, intermediate, and finish holds.
        """
        start_holds = self.filter_holds_by_type(detections, 'start_hold')
        intermediate_holds = self.filter_holds_by_type(detections, 'intermediate_hold')
        finish_holds = self.filter_holds_by_type(detections, 'finish_hold')
        volumes = self.filter_holds_by_type(detections, 'volume')
        
        return {
            'start_holds': start_holds,
            'intermediate_holds': intermediate_holds,
            'finish_holds': finish_holds,
            'volumes': volumes,
            'total_holds': len(detections['holds']),
            'has_start': len(start_holds) > 0,
            'has_finish': len(finish_holds) > 0
        }

