"""
Setup script for HoldOn project.
Creates necessary directories and checks dependencies.
"""

from pathlib import Path
import sys


def create_directories():
    """Create necessary project directories."""
    directories = [
        "models",
        "data",
        "data/roboflow",
        "data/kaggle",
        "runs",
        "outputs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'torch',
        'torchvision',
        'ultralytics',
        'cv2',
        'numpy',
        'PIL',
        'fastapi',
        'networkx',
        'scipy',
        'sklearn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} is NOT installed")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    return True


if __name__ == "__main__":
    print("🚀 Setting up HoldOn project...\n")
    
    print("📁 Creating directories...")
    create_directories()
    
    print("\n🔍 Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\n" + "="*50)
    if deps_ok:
        print("✅ Setup complete! You're ready to go.")
        print("\nNext steps:")
        print("1. Download datasets: python scripts/download_datasets.py")
        print("2. Train model: python src/training/train_hold_detector.py")
        print("3. Run example: python scripts/example_usage.py --image <path>")
        print("4. Start API: python -m src.api.main")
    else:
        print("⚠️  Setup incomplete. Please install missing dependencies.")
    print("="*50)

