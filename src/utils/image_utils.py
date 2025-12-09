"""
Image Processing Utilities
Helper functions for image preprocessing and augmentation.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from albumentations import (
    Compose, HorizontalFlip, Rotate, RandomBrightnessContrast,
    CLAHE, GaussNoise, Blur
)


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Preprocess image for model inference.
    
    Args:
        image: Input image (BGR format)
        target_size: Target size (width, height)
        
    Returns:
        Preprocessed image
    """
    # Resize while maintaining aspect ratio
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to target size
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    
    padded = cv2.copyMakeBorder(
        resized, pad_h, target_h - new_h - pad_h,
        pad_w, target_w - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=[114, 114, 114]
    )
    
    return padded


def get_augmentation_pipeline(mode: str = 'train') -> Compose:
    """
    Get data augmentation pipeline.
    
    Args:
        mode: 'train' or 'val'
        
    Returns:
        Albumentations compose object
    """
    if mode == 'train':
        return Compose([
            HorizontalFlip(p=0.5),
            Rotate(limit=15, p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            Blur(blur_limit=3, p=0.2)
        ], bbox_params=None)  # Add bbox_params if using bounding boxes
    else:
        return Compose([])  # No augmentation for validation


def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Enhance image for better hold detection.
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

