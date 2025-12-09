"""
Route Analysis Module
Analyzes climbing routes to assess difficulty, movement patterns, and spatial relationships.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler


class RouteAnalyzer:
    """
    Analyzes bouldering routes to determine difficulty, movement patterns,
    and optimal climbing sequences.
    """
    
    def __init__(self):
        """Initialize the route analyzer."""
        pass
    
    def calculate_hold_distances(self, holds: List[Dict]) -> np.ndarray:
        """
        Calculate pairwise distances between all holds.
        
        Args:
            holds: List of hold dictionaries with 'center' keys
            
        Returns:
            Distance matrix (n x n) where n is number of holds
        """
        n = len(holds)
        if n == 0:
            return np.array([])
        
        centers = np.array([[h['center']['x'], h['center']['y']] for h in holds])
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = euclidean(centers[i], centers[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def analyze_route_difficulty(self, detections: Dict, route_structure: Dict) -> Dict:
        """
        Analyze route difficulty based on hold positions, distances, and route structure.
        
        Args:
            detections: Detection results from HoldDetector
            route_structure: Route structure from HoldDetector.get_route_structure()
            
        Returns:
            Dictionary with difficulty metrics and assessment
        """
        all_holds = detections['holds']
        
        if len(all_holds) < 2:
            return {
                'difficulty_score': 0.0,
                'difficulty_grade': 'Unknown',
                'metrics': {}
            }
        
        # Calculate spatial metrics
        distances = self.calculate_hold_distances(all_holds)
        
        # Average distance between holds
        if distances.size > 0:
            avg_distance = np.mean(distances[distances > 0])
            max_distance = np.max(distances)
            min_distance = np.min(distances[distances > 0])
        else:
            avg_distance = max_distance = min_distance = 0
        
        # Vertical spread (how much the route goes up)
        y_coords = [h['center']['y'] for h in all_holds]
        vertical_spread = max(y_coords) - min(y_coords) if y_coords else 0
        
        # Horizontal spread
        x_coords = [h['center']['x'] for h in all_holds]
        horizontal_spread = max(x_coords) - min(x_coords) if x_coords else 0
        
        # Route complexity (number of holds)
        num_holds = len(all_holds)
        
        # Calculate difficulty score (0-100)
        # Factors: hold count, reach distances, vertical/horizontal spread
        difficulty_score = 0.0
        
        # More holds = more complex
        difficulty_score += min(num_holds * 2, 30)
        
        # Larger average distance = more difficult
        if avg_distance > 0:
            normalized_avg_dist = min(avg_distance / 200, 1.0)  # Normalize
            difficulty_score += normalized_avg_dist * 25
        
        # Vertical spread indicates route height
        if vertical_spread > 0:
            normalized_vert = min(vertical_spread / 500, 1.0)
            difficulty_score += normalized_vert * 20
        
        # Horizontal spread indicates route width/dynamism
        if horizontal_spread > 0:
            normalized_horz = min(horizontal_spread / 300, 1.0)
            difficulty_score += normalized_horz * 15
        
        # Route structure complexity
        if route_structure['has_start'] and route_structure['has_finish']:
            difficulty_score += 10
        
        difficulty_score = min(difficulty_score, 100)
        
        # Map to climbing grades (V-scale for bouldering)
        grade = self._score_to_grade(difficulty_score)
        
        return {
            'difficulty_score': float(difficulty_score),
            'difficulty_grade': grade,
            'metrics': {
                'num_holds': num_holds,
                'avg_hold_distance': float(avg_distance),
                'max_hold_distance': float(max_distance),
                'min_hold_distance': float(min_distance),
                'vertical_spread': float(vertical_spread),
                'horizontal_spread': float(horizontal_spread),
                'route_height': float(vertical_spread),
                'route_width': float(horizontal_spread)
            }
        }
    
    def _score_to_grade(self, score: float) -> str:
        """Convert difficulty score to bouldering grade (V-scale)."""
        if score < 20:
            return 'V0'
        elif score < 30:
            return 'V1'
        elif score < 40:
            return 'V2'
        elif score < 50:
            return 'V3'
        elif score < 60:
            return 'V4'
        elif score < 70:
            return 'V5'
        elif score < 80:
            return 'V6'
        elif score < 85:
            return 'V7'
        elif score < 90:
            return 'V8'
        elif score < 95:
            return 'V9'
        else:
            return 'V10+'
    
    def identify_movement_patterns(self, detections: Dict) -> Dict:
        """
        Identify movement patterns in the route (dynos, traverses, technical sections).
        
        Args:
            detections: Detection results from HoldDetector
            
        Returns:
            Dictionary with identified movement patterns
        """
        holds = detections['holds']
        
        if len(holds) < 2:
            return {
                'patterns': [],
                'dyno_sections': [],
                'technical_sections': [],
                'traverse_sections': []
            }
        
        distances = self.calculate_hold_distances(holds)
        centers = np.array([[h['center']['x'], h['center']['y']] for h in holds])
        
        patterns = []
        dyno_sections = []
        technical_sections = []
        traverse_sections = []
        
        # Identify potential dynos (large horizontal/vertical jumps)
        for i in range(len(holds) - 1):
            dist = distances[i, i + 1]
            dx = abs(centers[i + 1][0] - centers[i][0])
            dy = centers[i][1] - centers[i + 1][1]  # Negative = upward
            
            # Dyno: large distance with significant horizontal component
            if dist > 150 and dx > 50:
                dyno_sections.append({
                    'from_hold': i,
                    'to_hold': i + 1,
                    'distance': float(dist),
                    'type': 'dyno'
                })
                patterns.append('dyno')
            
            # Technical: small holds, close together
            elif dist < 80 and holds[i].get('area', 0) < 500:
                technical_sections.append({
                    'from_hold': i,
                    'to_hold': i + 1,
                    'distance': float(dist),
                    'type': 'technical'
                })
                patterns.append('technical')
            
            # Traverse: primarily horizontal movement
            elif dx > dy and dx > 100:
                traverse_sections.append({
                    'from_hold': i,
                    'to_hold': i + 1,
                    'distance': float(dist),
                    'type': 'traverse'
                })
                patterns.append('traverse')
        
        return {
            'patterns': list(set(patterns)),
            'dyno_sections': dyno_sections,
            'technical_sections': technical_sections,
            'traverse_sections': traverse_sections,
            'pattern_diversity': len(set(patterns))
        }
    
    def cluster_holds_by_position(self, holds: List[Dict], n_clusters: Optional[int] = None) -> Dict:
        """
        Cluster holds by spatial position to identify route sections.
        
        Args:
            holds: List of hold dictionaries
            n_clusters: Number of clusters (auto-determined if None)
            
        Returns:
            Dictionary with cluster assignments
        """
        if len(holds) < 2:
            return {'clusters': [], 'n_clusters': 0}
        
        centers = np.array([[h['center']['x'], h['center']['y']] for h in holds])
        
        # Auto-determine clusters based on route structure
        if n_clusters is None:
            # Use hierarchical clustering to find natural groupings
            if len(centers) > 1:
                linkage_matrix = linkage(centers, method='ward')
                # Cut tree to get reasonable number of clusters
                n_clusters = min(3, len(holds) // 2) if len(holds) > 3 else len(holds)
                cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            else:
                cluster_labels = [1]
        else:
            linkage_matrix = linkage(centers, method='ward')
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Group holds by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        return {
            'clusters': clusters,
            'n_clusters': len(clusters),
            'cluster_labels': cluster_labels.tolist()
        }

