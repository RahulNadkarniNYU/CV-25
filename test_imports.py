"""
Test script to verify all imports work correctly.
Run this to check if the project is set up properly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all module imports."""
    print("Testing imports...\n")
    
    tests = [
        ("Computer Vision - Hold Detector", "src.cv.hold_detector", "HoldDetector"),
        ("Computer Vision - Route Analyzer", "src.cv.route_analyzer", "RouteAnalyzer"),
        ("Computer Vision - Beta Generator", "src.cv.beta_generator", "BetaGenerator"),
        ("API - Main", "src.api.main", "app"),
        ("Training - Trainer", "src.training.train_hold_detector", "train_hold_detector"),
        ("Utils - Image Utils", "src.utils.image_utils", "preprocess_image"),
    ]
    
    passed = 0
    failed = 0
    
    for name, module_name, class_name in tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {name}")
            passed += 1
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed += 1
        except AttributeError as e:
            print(f"❌ {name}: Class/function not found - {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All imports successful! Project is ready to use.")
        return True
    else:
        print("⚠️  Some imports failed. Check dependencies.")
        return False


if __name__ == "__main__":
    test_imports()

