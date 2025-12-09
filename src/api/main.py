"""
FastAPI Server for HoldOn Bouldering Analysis
Provides REST API for route analysis and beta generation.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import io
from typing import Optional
import uvicorn

from src.cv.hold_detector import HoldDetector
from src.cv.route_analyzer import RouteAnalyzer
from src.cv.beta_generator import BetaGenerator

app = FastAPI(title="HoldOn API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models (lazy loading)
hold_detector = None
route_analyzer = None
beta_generator = None


def get_models():
    """Lazy load models on first request."""
    global hold_detector, route_analyzer, beta_generator
    
    if hold_detector is None:
        # Load model (update path to your trained model)
        # If model doesn't exist, will use pretrained COCO weights
        model_path = "models/hold_detector.pt"
        import os
        if not os.path.exists(model_path):
            model_path = None  # Use pretrained weights
        hold_detector = HoldDetector(model_path=model_path)
    
    if route_analyzer is None:
        route_analyzer = RouteAnalyzer()
    
    if beta_generator is None:
        beta_generator = BetaGenerator()
    
    return hold_detector, route_analyzer, beta_generator


def image_to_numpy(file: UploadFile) -> np.ndarray:
    """Convert uploaded image file to numpy array."""
    contents = file.file.read()
    image = Image.open(io.BytesIO(contents))
    image = image.convert('RGB')
    image_array = np.array(image)
    # Convert RGB to BGR for OpenCV
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    return image_array


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "HoldOn API - Bouldering Route Analysis", "status": "running"}


@app.post("/analyze-route")
async def analyze_route(
    image: UploadFile = File(...),
    user_height: Optional[int] = None,
    user_reach: Optional[int] = None,
    skill_level: Optional[str] = None
):
    """
    Analyze a bouldering route from an image.
    
    Returns:
        - Detected holds
        - Route structure
        - Difficulty analysis
        - Movement patterns
        - Beta sequences
        - Technique recommendations
    """
    try:
        # Load models
        detector, analyzer, generator = get_models()
        
        # Process image
        image_array = image_to_numpy(image)
        
        # Detect holds
        detections = detector.detect_holds(image_array)
        
        # Get route structure
        route_structure = detector.get_route_structure(detections)
        
        # Analyze route
        difficulty_analysis = analyzer.analyze_route_difficulty(detections, route_structure)
        movement_patterns = analyzer.identify_movement_patterns(detections)
        
        # Generate beta
        user_profile = {
            'height': user_height or 170,
            'reach': user_reach or 180,
            'skill_level': skill_level or 'intermediate',
            'style': 'balanced'
        }
        
        beta_results = generator.generate_beta(detections, route_structure, user_profile)
        
        # Generate technique recommendations
        if beta_results['primary_beta']:
            technique_recs = generator.generate_technique_recommendations(
                beta_results['primary_beta'],
                difficulty_analysis,
                user_profile
            )
        else:
            technique_recs = []
        
        return JSONResponse({
            "success": True,
            "detections": {
                "num_holds": detections['num_holds'],
                "holds": [
                    {
                        "id": h['id'],
                        "class": h['class'],
                        "confidence": h['confidence'],
                        "center": h['center'],
                        "bbox": h['bbox']
                    }
                    for h in detections['holds']
                ]
            },
            "route_structure": {
                "has_start": route_structure['has_start'],
                "has_finish": route_structure['has_finish'],
                "total_holds": route_structure['total_holds'],
                "start_holds": len(route_structure['start_holds']),
                "intermediate_holds": len(route_structure['intermediate_holds']),
                "finish_holds": len(route_structure['finish_holds'])
            },
            "difficulty": difficulty_analysis,
            "movement_patterns": movement_patterns,
            "beta": {
                "primary_sequence": [
                    {
                        "id": h['id'],
                        "class": h['class'],
                        "center": h['center']
                    }
                    for h in beta_results['primary_beta']
                ],
                "num_alternatives": beta_results['num_alternatives']
            },
            "technique_recommendations": technique_recs
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing route: {str(e)}")


@app.post("/detect-holds")
async def detect_holds(image: UploadFile = File(...), conf_threshold: float = 0.25):
    """
    Detect holds in an image.
    
    Returns:
        - List of detected holds with bounding boxes and classes
    """
    try:
        detector, _, _ = get_models()
        image_array = image_to_numpy(image)
        
        detections = detector.detect_holds(image_array, conf_threshold=conf_threshold)
        
        return JSONResponse({
            "success": True,
            "num_holds": detections['num_holds'],
            "holds": [
                {
                    "id": h['id'],
                    "class": h['class'],
                    "confidence": h['confidence'],
                    "center": h['center'],
                    "bbox": h['bbox']
                }
                for h in detections['holds']
            ]
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting holds: {str(e)}")


@app.post("/visualize-route")
async def visualize_route(image: UploadFile = File(...), conf_threshold: float = 0.25):
    """
    Visualize detected holds on the image.
    
    Returns:
        - Base64 encoded image with visualizations
    """
    try:
        import base64
        
        detector, _, _ = get_models()
        image_array = image_to_numpy(image)
        
        detections = detector.detect_holds(image_array, conf_threshold=conf_threshold)
        vis_image = detector.visualize_detections(image_array, detections)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', vis_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse({
            "success": True,
            "image": f"data:image/jpeg;base64,{image_base64}",
            "num_holds": detections['num_holds']
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error visualizing route: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

