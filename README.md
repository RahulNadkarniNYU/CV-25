# HoldOn - Bouldering Route Analysis System

A computer vision and AI-powered system for analyzing bouldering problems and generating climbing beta (sequence suggestions).

## 🎯 Features

### Core Functionality
- **Hold Detection**: Detects and classifies climbing holds (start, intermediate, finish, volumes) using YOLOv8 segmentation
- **Route Analysis**: Analyzes route difficulty, movement patterns, and spatial relationships
- **Beta Generation**: Generates optimal climbing sequences based on route structure and user profile
- **Technique Recommendations**: Provides technique suggestions for each move

### Secondary Features (Planned)
- Community-driven route database
- Personal climbing progress tracking
- Route recommendations
- Social features
- Gym/crag location integration

## 🏗️ Project Structure

```
HoldOn/
├── src/
│   ├── cv/                    # Computer Vision Module
│   │   ├── hold_detector.py   # Hold detection using YOLOv8
│   │   ├── route_analyzer.py  # Route difficulty and pattern analysis
│   │   └── beta_generator.py  # Beta sequence generation
│   ├── api/                   # API Server
│   │   └── main.py           # FastAPI REST API
│   ├── training/              # Model Training
│   │   └── train_hold_detector.py
│   └── utils/                 # Utilities
│       └── image_utils.py
├── scripts/
│   ├── download_datasets.py   # Dataset download scripts
│   └── example_usage.py       # Example usage
├── models/                    # Trained models (create this directory)
├── data/                      # Datasets (create this directory)
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

1. **Clone the repository** (or navigate to project directory)

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download datasets** (optional, for training):
```bash
# Roboflow dataset
python scripts/download_datasets.py --roboflow-key YOUR_API_KEY

# Kaggle dataset (requires Kaggle API setup)
python scripts/download_datasets.py --kaggle-dataset tomasslama/indoor-climbing-gym-hold-segmentation
```

### Usage

#### 1. Basic Route Analysis (Python)

```python
from src.cv.hold_detector import HoldDetector
from src.cv.route_analyzer import RouteAnalyzer
from src.cv.beta_generator import BetaGenerator
import cv2

# Load image
image = cv2.imread("path/to/bouldering_wall.jpg")

# Initialize components
detector = HoldDetector(model_path="models/hold_detector.pt")  # Use trained model
analyzer = RouteAnalyzer()
generator = BetaGenerator()

# Detect holds
detections = detector.detect_holds(image)

# Analyze route
route_structure = detector.get_route_structure(detections)
difficulty = analyzer.analyze_route_difficulty(detections, route_structure)

# Generate beta
user_profile = {
    'height': 175,  # cm
    'reach': 185,   # cm
    'skill_level': 'intermediate',
    'style': 'balanced'
}
beta = generator.generate_beta(detections, route_structure, user_profile)

# Visualize
vis_image = detector.visualize_detections(image, detections)
cv2.imwrite("output.jpg", vis_image)
```

#### 2. Command Line Example

```bash
python scripts/example_usage.py --image path/to/image.jpg --model models/hold_detector.pt
```

#### 3. API Server

```bash
# Start the API server
python -m src.api.main

# Or using uvicorn directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**API Endpoints:**
- `POST /analyze-route`: Full route analysis with beta generation
- `POST /detect-holds`: Hold detection only
- `POST /visualize-route`: Visualized detection results

**Example API Request:**
```bash
curl -X POST "http://localhost:8000/analyze-route" \
  -F "image=@path/to/image.jpg" \
  -F "user_height=175" \
  -F "user_reach=185" \
  -F "skill_level=intermediate"
```

## 🎓 Model Training

### Training Hold Detection Model

1. **Prepare dataset**:
   - Download datasets using the scripts
   - Organize in YOLOv8 format:
     ```
     dataset/
     ├── train/
     │   ├── images/
     │   └── labels/
     ├── valid/
     │   ├── images/
     │   └── labels/
     └── test/
         ├── images/
         └── labels/
     ```

2. **Create dataset config** (YAML):
```yaml
path: /path/to/dataset
train: train/images
val: valid/images
test: test/images
nc: 4
names:
  0: start_hold
  1: intermediate_hold
  2: finish_hold
  3: volume
```

3. **Train model**:
```bash
python src/training/train_hold_detector.py \
  --data dataset_config.yaml \
  --model-size n \
  --epochs 100 \
  --batch 16
```

### Model Options
- `--model-size`: n (nano), s (small), m (medium), l (large), x (xlarge)
- `--epochs`: Number of training epochs
- `--batch`: Batch size
- `--imgsz`: Image size (default: 640)

## 📊 Computer Vision Pipeline

### 1. Hold Detection
- **Model**: YOLOv8 Segmentation
- **Classes**: Start holds, Intermediate holds, Finish holds, Volumes
- **Output**: Bounding boxes, segmentation masks, confidence scores

### 2. Route Analysis
- **Spatial Analysis**: Hold distances, route spread, clustering
- **Difficulty Assessment**: Multi-factor scoring (hold count, distances, spread)
- **Movement Patterns**: Dyno detection, technical sections, traverses

### 3. Beta Generation
- **Pathfinding**: Graph-based optimal path finding (Dijkstra's algorithm)
- **User Adaptation**: Considers height, reach, skill level
- **Technique Recommendations**: Move-specific technique suggestions

## 🔧 Configuration

### Model Paths
Update model paths in:
- `src/api/main.py`: API model loading
- `src/cv/hold_detector.py`: Default model initialization

### User Profile
Customize user profile parameters:
```python
user_profile = {
    'height': 175,        # cm
    'reach': 185,         # cm (wingspan)
    'skill_level': 'beginner' | 'intermediate' | 'advanced' | 'expert',
    'style': 'balanced' | 'dynamic' | 'static'
}
```

## 📈 Future Enhancements

### Computer Vision
- [ ] 3D pose estimation for movement analysis
- [ ] Real-time video analysis
- [ ] Multi-angle route reconstruction
- [ ] Hold type classification (crimps, jugs, slopers, etc.)

### Beta Generation
- [ ] Machine learning-based beta ranking
- [ ] Style-specific beta generation
- [ ] Dynamic movement prediction
- [ ] Injury risk assessment

### System Features
- [ ] Mobile app integration
- [ ] Cloud model serving
- [ ] Model quantization for edge devices
- [ ] Database integration for route storage

## 📚 Datasets

### Roboflow Dataset
- **Link**: https://universe.roboflow.com/boulderingdataset/climbing-holds-and-volumes-dohxi
- **Format**: YOLOv8
- **Classes**: Climbing holds and volumes

### Kaggle Dataset
- **Link**: https://www.kaggle.com/datasets/tomasslama/indoor-climbing-gym-hold-segmentation
- **Format**: Segmentation masks
- **Focus**: Indoor climbing gym holds

## 🤝 Contributing

This is a computer vision-focused project. Key areas for contribution:
- Model improvements and training
- Dataset collection and annotation
- Algorithm optimization
- Mobile deployment

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- Roboflow for the climbing holds dataset
- Kaggle community for the indoor climbing gym dataset
- Ultralytics for YOLOv8

---

**Note**: This project requires a trained model for optimal performance. Use the training scripts with the provided datasets to train your own model, or fine-tune from pretrained weights.

