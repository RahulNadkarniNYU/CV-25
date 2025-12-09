"""
Example usage of HoldOn computer vision pipeline.
"""

import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.cv.hold_detector import HoldDetector
from src.cv.route_analyzer import RouteAnalyzer
from src.cv.beta_generator import BetaGenerator


def analyze_route_example(image_path: str, model_path: str = None):
    """
    Example: Analyze a bouldering route from an image.
    
    Args:
        image_path: Path to input image
        model_path: Path to trained model (optional)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print("🔍 Initializing models...")
    
    # Initialize components
    detector = HoldDetector(model_path=model_path)
    analyzer = RouteAnalyzer()
    generator = BetaGenerator()
    
    print("📸 Detecting holds...")
    
    # Detect holds (lower confidence for testing)
    detections = detector.detect_holds(image, conf_threshold=0.1)
    print(f"   Found {detections['num_holds']} holds")
    
    # Get route structure
    route_structure = detector.get_route_structure(detections)
    print(f"   Start holds: {len(route_structure['start_holds'])}")
    print(f"   Intermediate holds: {len(route_structure['intermediate_holds'])}")
    print(f"   Finish holds: {len(route_structure['finish_holds'])}")
    
    print("\n📊 Analyzing route difficulty...")
    
    # Analyze route
    difficulty = analyzer.analyze_route_difficulty(detections, route_structure)
    print(f"   Difficulty Score: {difficulty['difficulty_score']:.1f}/100")
    print(f"   Grade: {difficulty['difficulty_grade']}")
    if difficulty['metrics'] and 'avg_hold_distance' in difficulty['metrics']:
        print(f"   Average hold distance: {difficulty['metrics']['avg_hold_distance']:.1f} pixels")
    
    # Movement patterns
    patterns = analyzer.identify_movement_patterns(detections)
    print(f"\n🎯 Movement Patterns:")
    print(f"   Patterns found: {', '.join(patterns['patterns']) if patterns['patterns'] else 'None'}")
    print(f"   Dyno sections: {len(patterns['dyno_sections'])}")
    print(f"   Technical sections: {len(patterns['technical_sections'])}")
    
    print("\n🧗 Generating beta...")
    
    # Generate beta
    user_profile = {
        'height': 175,  # cm
        'reach': 185,   # cm
        'skill_level': 'intermediate',
        'style': 'balanced'
    }
    
    beta = generator.generate_beta(detections, route_structure, user_profile)
    print(f"   Primary beta: {len(beta['primary_beta'])} moves")
    print(f"   Alternative betas: {beta.get('num_alternatives', 0)}")
    
    # Technique recommendations
    if beta['primary_beta']:
        techniques = generator.generate_technique_recommendations(
            beta['primary_beta'], difficulty, user_profile
        )
        print(f"\n💡 Technique Recommendations:")
        for rec in techniques[:5]:  # Show first 5
            print(f"   Move {rec['move_number']}: {', '.join([t['technique'] for t in rec['techniques']])}")
    
    print("\n🎨 Visualizing detections...")
    
    # Visualize
    vis_image = detector.visualize_detections(image, detections)
    
    # Save output
    output_path = "output_analysis.jpg"
    cv2.imwrite(output_path, vis_image)
    print(f"   Saved visualization to: {output_path}")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Example HoldOn usage")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    
    args = parser.parse_args()
    
    analyze_route_example(args.image, args.model)

