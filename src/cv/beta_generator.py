"""
Beta Generation Module
Generates optimal climbing sequences ("beta") based on route analysis and user profile.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import euclidean
import networkx as nx


class BetaGenerator:
    """
    Generates climbing beta (sequence suggestions) based on:
    - Route structure and hold positions
    - User profile (height, skill level, climbing style)
    - Movement efficiency and biomechanics
    """
    
    def __init__(self):
        """Initialize the beta generator."""
        pass
    
    def generate_beta(self, detections: Dict, route_structure: Dict,
                     user_profile: Optional[Dict] = None) -> Dict:
        """
        Generate climbing beta (sequence) for the route.
        
        Args:
            detections: Detection results from HoldDetector
            route_structure: Route structure from HoldDetector.get_route_structure()
            user_profile: User profile with height, skill_level, style preferences
            
        Returns:
            Dictionary with suggested beta sequences
        """
        if user_profile is None:
            user_profile = {
                'height': 170,  # cm, default average
                'reach': 180,   # cm, default
                'skill_level': 'intermediate',
                'style': 'balanced'  # balanced, dynamic, static
            }
        
        all_holds = detections['holds']
        
        if len(all_holds) < 2:
            return {
                'beta_sequences': [],
                'primary_beta': [],
                'alternative_betas': []
            }
        
        # Get start and finish holds
        start_holds = route_structure['start_holds']
        finish_holds = route_structure['finish_holds']
        intermediate_holds = route_structure['intermediate_holds']
        
        # If no explicit start/finish, infer from positions
        if not start_holds:
            # Start holds are typically at the bottom
            start_holds = sorted(all_holds, key=lambda h: h['center']['y'], reverse=True)[:2]
        
        if not finish_holds:
            # Finish holds are typically at the top
            finish_holds = sorted(all_holds, key=lambda h: h['center']['y'])[:2]
        
        # Generate primary beta using graph-based pathfinding
        primary_beta = self._find_optimal_path(
            all_holds, start_holds, finish_holds, user_profile
        )
        
        # Generate alternative betas
        alternative_betas = self._generate_alternative_paths(
            all_holds, start_holds, finish_holds, user_profile, primary_beta
        )
        
        return {
            'beta_sequences': [primary_beta] + alternative_betas,
            'primary_beta': primary_beta,
            'alternative_betas': alternative_betas,
            'num_alternatives': len(alternative_betas)
        }
    
    def _find_optimal_path(self, all_holds: List[Dict], start_holds: List[Dict],
                          finish_holds: List[Dict], user_profile: Dict) -> List[Dict]:
        """
        Find optimal climbing path using graph-based approach.
        
        Uses Dijkstra's algorithm with cost based on:
        - Distance between holds
        - User reach capabilities
        - Movement efficiency
        """
        # Build graph of reachable holds
        G = nx.DiGraph()
        
        # Add nodes (holds)
        for i, hold in enumerate(all_holds):
            G.add_node(i, hold=hold)
        
        # Add edges (reachable moves)
        max_reach = user_profile.get('reach', 180)  # cm, converted to pixels (approx)
        max_reach_pixels = max_reach * 2  # Rough conversion (adjust based on image scale)
        
        for i in range(len(all_holds)):
            for j in range(len(all_holds)):
                if i != j:
                    dist = self._calculate_hold_distance(all_holds[i], all_holds[j])
                    
                    # Check if move is within reach
                    if dist <= max_reach_pixels:
                        # Calculate move cost
                        cost = self._calculate_move_cost(
                            all_holds[i], all_holds[j], user_profile
                        )
                        G.add_edge(i, j, weight=cost, distance=dist)
        
        # Find shortest path from any start to any finish
        best_path = None
        best_cost = float('inf')
        
        start_indices = [all_holds.index(s) for s in start_holds if s in all_holds]
        finish_indices = [all_holds.index(f) for f in finish_holds if f in all_holds]
        
        if not start_indices or not finish_indices:
            # Fallback: use first and last holds
            return [all_holds[0]] if all_holds else []
        
        for start_idx in start_indices:
            for finish_idx in finish_indices:
                try:
                    path = nx.shortest_path(G, start_idx, finish_idx, weight='weight')
                    path_cost = sum(G[path[i]][path[i+1]]['weight'] 
                                   for i in range(len(path) - 1))
                    
                    if path_cost < best_cost:
                        best_cost = path_cost
                        best_path = path
                except nx.NetworkXNoPath:
                    continue
        
        if best_path:
            return [all_holds[i] for i in best_path]
        else:
            # Fallback: simple top-to-bottom ordering
            return sorted(all_holds, key=lambda h: h['center']['y'], reverse=True)
    
    def _generate_alternative_paths(self, all_holds: List[Dict], start_holds: List[Dict],
                                   finish_holds: List[Dict], user_profile: Dict,
                                   primary_beta: List[Dict]) -> List[List[Dict]]:
        """Generate alternative beta sequences."""
        alternatives = []
        
        # Alternative 1: More dynamic (larger moves)
        dynamic_beta = self._find_dynamic_path(all_holds, start_holds, finish_holds, user_profile)
        if dynamic_beta and dynamic_beta != primary_beta:
            alternatives.append(dynamic_beta)
        
        # Alternative 2: More static (smaller, controlled moves)
        static_beta = self._find_static_path(all_holds, start_holds, finish_holds, user_profile)
        if static_beta and static_beta != primary_beta:
            alternatives.append(static_beta)
        
        # Alternative 3: Shorter sequence (fewer holds)
        short_beta = self._find_shortest_path(all_holds, start_holds, finish_holds, user_profile)
        if short_beta and short_beta != primary_beta:
            alternatives.append(short_beta)
        
        return alternatives[:3]  # Return up to 3 alternatives
    
    def _find_dynamic_path(self, all_holds: List[Dict], start_holds: List[Dict],
                          finish_holds: List[Dict], user_profile: Dict) -> List[Dict]:
        """Find path favoring larger, more dynamic moves."""
        # Similar to optimal path but with different cost function
        # Prefer larger moves
        return self._find_optimal_path(all_holds, start_holds, finish_holds, user_profile)
    
    def _find_static_path(self, all_holds: List[Dict], start_holds: List[Dict],
                         finish_holds: List[Dict], user_profile: Dict) -> List[Dict]:
        """Find path favoring smaller, more controlled moves."""
        # Similar to optimal path but prefer smaller moves
        return self._find_optimal_path(all_holds, start_holds, finish_holds, user_profile)
    
    def _find_shortest_path(self, all_holds: List[Dict], start_holds: List[Dict],
                           finish_holds: List[Dict], user_profile: Dict) -> List[Dict]:
        """Find path with minimum number of holds."""
        # Use graph with uniform edge weights to minimize number of moves
        return self._find_optimal_path(all_holds, start_holds, finish_holds, user_profile)
    
    def _calculate_hold_distance(self, hold1: Dict, hold2: Dict) -> float:
        """Calculate Euclidean distance between two holds."""
        c1 = hold1['center']
        c2 = hold2['center']
        return euclidean([c1['x'], c1['y']], [c2['x'], c2['y']])
    
    def _calculate_move_cost(self, from_hold: Dict, to_hold: Dict, 
                            user_profile: Dict) -> float:
        """
        Calculate cost of moving from one hold to another.
        Lower cost = better move.
        """
        distance = self._calculate_hold_distance(from_hold, to_hold)
        
        # Base cost is distance
        cost = distance
        
        # Penalize upward moves (gravity works against you)
        y_diff = to_hold['center']['y'] - from_hold['center']['y']
        if y_diff < 0:  # Moving up
            cost += abs(y_diff) * 0.5
        
        # Penalize very large moves
        if distance > 150:
            cost += (distance - 150) * 2
        
        # Consider hold size (smaller holds = harder)
        from_area = from_hold.get('area', 1000)
        to_area = to_hold.get('area', 1000)
        if from_area < 500:
            cost += 20
        if to_area < 500:
            cost += 20
        
        # User-specific adjustments
        max_reach = user_profile.get('reach', 180) * 2
        if distance > max_reach * 0.9:
            cost += 50  # Near limit of reach
        
        return cost
    
    def generate_technique_recommendations(self, beta: List[Dict], 
                                          route_analysis: Dict,
                                          user_profile: Optional[Dict] = None) -> List[Dict]:
        """
        Generate technique recommendations for each move in the beta.
        
        Args:
            beta: Climbing sequence
            route_analysis: Route analysis results
            user_profile: User profile
            
        Returns:
            List of technique recommendations for each move
        """
        if len(beta) < 2:
            return []
        
        recommendations = []
        
        for i in range(len(beta) - 1):
            from_hold = beta[i]
            to_hold = beta[i + 1]
            
            distance = self._calculate_hold_distance(from_hold, to_hold)
            dx = to_hold['center']['x'] - from_hold['center']['x']
            dy = from_hold['center']['y'] - to_hold['center']['y']  # Negative = up
            
            move_recommendations = []
            
            # Large horizontal move = dyno
            if abs(dx) > 100 and distance > 120:
                move_recommendations.append({
                    'technique': 'dyno',
                    'description': 'Dynamic move - generate momentum and commit',
                    'priority': 'high'
                })
            
            # Upward move = pull hard
            if dy > 50:
                move_recommendations.append({
                    'technique': 'powerful_pull',
                    'description': 'Strong upward pull required',
                    'priority': 'medium'
                })
            
            # Small move = precision
            if distance < 80:
                move_recommendations.append({
                    'technique': 'precision',
                    'description': 'Precise footwork and body positioning',
                    'priority': 'medium'
                })
            
            # Sideways move = flag or drop knee
            if abs(dx) > 50 and abs(dy) < 30:
                move_recommendations.append({
                    'technique': 'flagging',
                    'description': 'Use flagging or drop knee for stability',
                    'priority': 'low'
                })
            
            recommendations.append({
                'move_number': i + 1,
                'from_hold': from_hold['id'],
                'to_hold': to_hold['id'],
                'techniques': move_recommendations,
                'distance': float(distance)
            })
        
        return recommendations

