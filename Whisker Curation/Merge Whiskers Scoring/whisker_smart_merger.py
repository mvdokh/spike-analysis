"""
Smart Whisker Merger Script
Uses multiple criteria to intelligently identify which whiskers should be merged.
Distinguishes between split whiskers and noise.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import os


def parse_coordinates(coord_string):
    """Parse coordinate string into list of floats."""
    if pd.isna(coord_string) or coord_string == '':
        return []
    return [float(x.strip()) for x in str(coord_string).split(',')]


def get_whisker_endpoints(x_coords, y_coords):
    """Get the base (first point) and tip (last point) of a whisker."""
    if len(x_coords) < 2 or len(y_coords) < 2:
        return None, None
    
    base = (x_coords[0], y_coords[0])
    tip = (x_coords[-1], y_coords[-1])
    
    return base, tip


def calculate_whisker_angle(x_coords, y_coords):
    """Calculate the overall angle/orientation of a whisker from base to tip."""
    if len(x_coords) < 2:
        return None
    
    dx = x_coords[-1] - x_coords[0]
    dy = y_coords[-1] - y_coords[0]
    
    return np.arctan2(dy, dx)


def calculate_whisker_length(x_coords, y_coords):
    """Calculate total length of whisker as sum of segment lengths."""
    if len(x_coords) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(len(x_coords) - 1):
        dx = x_coords[i+1] - x_coords[i]
        dy = y_coords[i+1] - y_coords[i]
        total_length += np.sqrt(dx**2 + dy**2)
    
    return total_length


def point_to_line_distance(point, line_start, line_end):
    """
    Calculate the perpendicular distance from a point to a line segment.
    
    Returns:
    --------
    float : Distance from point to line
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Vector from line_start to line_end
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        # Line start and end are the same point
        return euclidean(point, line_start)
    
    # Parameter t represents position along the line
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    
    # Clamp t to [0, 1] to stay on the line segment
    t = max(0, min(1, t))
    
    # Closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return euclidean(point, (closest_x, closest_y))


def calculate_continuation_score(w1, w2):
    """
    Check if extending one whisker's trajectory would intersect the other whisker.
    This is a strong indicator that they are parts of the same split whisker.
    
    Returns:
    --------
    dict : {'score': float, 'w1_continues_to_w2': bool, 'w2_continues_to_w1': bool, 'min_distance': float}
    """
    # Check if w1's trajectory (from base to tip direction) would hit w2
    base1, tip1 = w1['base'], w1['tip']
    base2, tip2 = w2['base'], w2['tip']
    
    if base1 is None or base2 is None:
        return {'score': 0, 'w1_continues_to_w2': False, 'w2_continues_to_w1': False, 'min_distance': float('inf')}
    
    # Get whisker directions
    angle1 = w1['angle']
    angle2 = w2['angle']
    
    if angle1 is None or angle2 is None:
        return {'score': 0, 'w1_continues_to_w2': False, 'w2_continues_to_w1': False, 'min_distance': float('inf')}
    
    score = 0
    w1_continues = False
    w2_continues = False
    min_dist = float('inf')
    
    # Extend w1's tip in its direction and check distance to w2's points
    extension_length = 100  # pixels
    extended_tip1_x = tip1[0] + extension_length * np.cos(angle1)
    extended_tip1_y = tip1[1] + extension_length * np.sin(angle1)
    
    # Check if any point on w2 is close to w1's extended trajectory
    for i in range(len(w2['x_coords'])):
        point = (w2['x_coords'][i], w2['y_coords'][i])
        dist = point_to_line_distance(point, tip1, (extended_tip1_x, extended_tip1_y))
        min_dist = min(min_dist, dist)
        
        if dist < 15:  # Within 15 pixels of extended line
            w1_continues = True
            break
    
    # Extend w2's tip in its direction and check distance to w1's points
    extended_tip2_x = tip2[0] + extension_length * np.cos(angle2)
    extended_tip2_y = tip2[1] + extension_length * np.sin(angle2)
    
    for i in range(len(w1['x_coords'])):
        point = (w1['x_coords'][i], w1['y_coords'][i])
        dist = point_to_line_distance(point, tip2, (extended_tip2_x, extended_tip2_y))
        min_dist = min(min_dist, dist)
        
        if dist < 15:
            w2_continues = True
            break
    
    # Score based on continuation
    if w1_continues or w2_continues:
        score = 50  # Strong indicator of split whisker
        if w1_continues and w2_continues:
            score = 70  # Even stronger - mutual continuation
    
    return {
        'score': score,
        'w1_continues_to_w2': w1_continues,
        'w2_continues_to_w1': w2_continues,
        'min_distance': min_dist
    }


def identify_fixture_points(whiskers, cluster_radius=30.0):
    """
    Identify whisker base "fixture points" that are clustered together.
    These are stable anchor points on the face that should never be merged.
    
    Returns:
    --------
    list : List of (x, y) tuples representing fixture point locations
    """
    # Collect all base points
    bases = [w['base'] for w in whiskers if w['base'] is not None]
    
    if len(bases) < 3:
        return []
    
    # Find clusters of bases (points close together)
    fixture_points = []
    
    for base in bases:
        # Check if this base is close to existing fixture points
        is_near_fixture = any(
            euclidean(base, fp) < cluster_radius 
            for fp in fixture_points
        )
        
        if is_near_fixture:
            continue
        
        # Count how many other bases are near this one
        nearby_count = sum(
            1 for other_base in bases 
            if euclidean(base, other_base) < cluster_radius
        )
        
        # If multiple bases cluster here, it's likely a fixture point
        if nearby_count >= 2:  # At least 2 other bases nearby (3 total)
            fixture_points.append(base)
    
    return fixture_points


def is_fixture_point(point, fixture_points, tolerance=15.0):
    """Check if a point is near a fixture point."""
    return any(euclidean(point, fp) < tolerance for fp in fixture_points)


def calculate_overlap_score(w1, w2):
    """
    Calculate how much two whiskers overlap or share similar spatial regions.
    Higher score means more overlap (potential split whisker).
    
    Returns:
    --------
    float : Overlap score (0-100+)
    """
    # Sample points along each whisker
    x1, y1 = w1['x_coords'], w1['y_coords']
    x2, y2 = w2['x_coords'], w2['y_coords']
    
    if len(x1) < 2 or len(x2) < 2:
        return 0.0
    
    # For each point in whisker1, find closest point in whisker2
    min_distances = []
    
    # Sample every few points to avoid too much computation
    step = max(1, len(x1) // 10)
    for i in range(0, len(x1), step):
        min_dist = min(
            euclidean((x1[i], y1[i]), (x2[j], y2[j]))
            for j in range(len(x2))
        )
        min_distances.append(min_dist)
    
    # If many points are very close, whiskers likely overlap
    avg_min_dist = np.mean(min_distances)
    close_points = sum(1 for d in min_distances if d < 20.0)
    close_ratio = close_points / len(min_distances)
    
    # Score based on proximity and ratio of close points
    overlap_score = 0
    if avg_min_dist < 30.0:
        overlap_score += 30
    if close_ratio > 0.3:
        overlap_score += 40 * close_ratio
    
    return overlap_score


def check_collinearity(w1, w2, angle_tolerance=np.pi/6):
    """
    Check if two whiskers are roughly collinear (pointing in similar directions).
    Returns True if they could be parts of the same whisker.
    """
    if w1['angle'] is None or w2['angle'] is None:
        return False
    
    angle_diff = abs(w1['angle'] - w2['angle'])
    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff
    
    return angle_diff < angle_tolerance


def find_min_distance_between_whiskers(x1, y1, x2, y2):
    """
    Find the minimum distance between any two points on two whiskers.
    
    Returns:
    --------
    tuple : (min_distance, idx1, idx2)
    """
    min_dist = float('inf')
    best_i, best_j = 0, 0
    
    for i in range(len(x1)):
        for j in range(len(x2)):
            dist = euclidean((x1[i], y1[i]), (x2[j], y2[j]))
            if dist < min_dist:
                min_dist = dist
                best_i, best_j = i, j
    
    return min_dist, best_i, best_j


def calculate_endpoint_distances(w1, w2):
    """Calculate all endpoint-to-endpoint distances."""
    base1, tip1 = w1['base'], w1['tip']
    base2, tip2 = w2['base'], w2['tip']
    
    if base1 is None or base2 is None:
        return {}
    
    return {
        'tip1-base2': euclidean(tip1, base2),
        'base1-tip2': euclidean(base1, tip2),
        'tip1-tip2': euclidean(tip1, tip2),
        'base1-base2': euclidean(base1, base2)
    }


def is_noise_whisker(whisker, all_whiskers, min_length_ratio=0.3):
    """
    Determine if a whisker is likely noise based on relative length.
    
    Parameters:
    -----------
    whisker : dict
        Whisker to check
    all_whiskers : list
        All whiskers in the frame
    min_length_ratio : float
        Minimum length ratio compared to median (default: 0.3 = 30%)
    
    Returns:
    --------
    bool : True if likely noise
    """
    lengths = [w['length'] for w in all_whiskers if w['length'] > 0]
    if not lengths:
        return False
    
    median_length = np.median(lengths)
    
    # If this whisker is much shorter than median, it's likely noise
    return whisker['length'] < (median_length * min_length_ratio)


def score_merge_candidate(w1, w2, distance_threshold=30.0, fixture_points=None):
    """
    Score how good of a merge candidate two whiskers are.
    Higher score = better candidate.
    
    Criteria (in order of importance):
    1. Fixture point protection (NEVER merge bases near fixture points)
    2. Collinearity (same direction = split whisker)
    3. Continuation (would extending one whisker hit the other?)
    4. Endpoint proximity (are endpoints close?)
    5. Overlap (do they occupy similar space?)
    
    Returns:
    --------
    dict : {'score': float, 'reason': str, 'merge_type': str, 'distance': float}
    """
    if fixture_points is None:
        fixture_points = []
    
    # Check basic validity
    if len(w1['x_coords']) < 2 or len(w2['x_coords']) < 2:
        return {'score': -1000, 'reason': 'invalid', 'merge_type': None, 'distance': float('inf')}
    
    score = 0
    reasons = []
    
    # ===== CRITICAL: Fixture point protection =====
    base1_is_fixture = is_fixture_point(w1['base'], fixture_points, tolerance=20.0)
    base2_is_fixture = is_fixture_point(w2['base'], fixture_points, tolerance=20.0)
    
    if base1_is_fixture and base2_is_fixture:
        # Both bases are at fixture points - absolutely do not merge!
        return {'score': -1000, 'reason': 'both_bases_at_fixture_points', 'merge_type': None, 'distance': float('inf')}
    
    # ===== 1. Collinearity (MOST IMPORTANT) =====
    # Split whiskers MUST point in the same direction
    is_collinear = check_collinearity(w1, w2, angle_tolerance=np.pi/4)  # 45 degrees
    if is_collinear:
        score += 100  # Massive boost - this is the primary indicator
        reasons.append('COLLINEAR')
    else:
        # Not collinear = likely different whiskers, heavy penalty
        score -= 150
        reasons.append('NOT_COLLINEAR')
        # If not collinear, likely not a split whisker - return early
        return {'score': score, 'reason': ', '.join(reasons), 'merge_type': None, 'distance': float('inf')}
    
    # ===== 2. Continuation (VERY IMPORTANT) =====
    # If extending one whisker's line would hit the other, they're likely split parts
    continuation_info = calculate_continuation_score(w1, w2)
    score += continuation_info['score']
    if continuation_info['w1_continues_to_w2'] and continuation_info['w2_continues_to_w1']:
        reasons.append('STRONG_CONTINUATION')
    elif continuation_info['w1_continues_to_w2'] or continuation_info['w2_continues_to_w1']:
        reasons.append('continuation')
    
    # ===== 3. Endpoint distances =====
    endpoint_dists = calculate_endpoint_distances(w1, w2)
    if not endpoint_dists:
        return {'score': -1000, 'reason': 'invalid', 'merge_type': None, 'distance': float('inf')}
    
    # Find best endpoint connection
    best_connection = min(endpoint_dists.items(), key=lambda x: x[1])
    merge_type, min_endpoint_dist = best_connection
    
    # Only consider if endpoints are reasonably close
    if min_endpoint_dist > distance_threshold:
        return {'score': -50, 'reason': 'too_far', 'merge_type': merge_type, 'distance': min_endpoint_dist}
    
    # Reward close endpoints
    if min_endpoint_dist < distance_threshold * 0.3:
        score += 40
        reasons.append('very_close_endpoints')
    elif min_endpoint_dist < distance_threshold * 0.6:
        score += 25
        reasons.append('close_endpoints')
    elif min_endpoint_dist < distance_threshold:
        score += 10
        reasons.append('moderate_endpoints')
    
    # ===== 4. Penalize connections involving fixture bases =====
    if 'base1' in merge_type and base1_is_fixture:
        score -= 200
        reasons.append('FIXTURE_BASE1_penalty')
    if 'base2' in merge_type and base2_is_fixture:
        score -= 200
        reasons.append('FIXTURE_BASE2_penalty')
    
    # ===== 5. Penalize unnatural connections =====
    if merge_type == 'base1-base2':
        score -= 80  # Stronger penalty - creates U-shapes
        reasons.append('BASE_TO_BASE_penalty')
    elif merge_type == 'tip1-tip2':
        score -= 80
        reasons.append('TIP_TO_TIP_penalty')
    else:
        score += 30  # Natural end-to-end connection
        reasons.append('natural_connection')
    
    # ===== 6. Overlap/similarity (helpful but not critical) =====
    overlap = calculate_overlap_score(w1, w2)
    if overlap > 30:
        score += 20
        reasons.append(f'overlap({overlap:.0f})')
    elif overlap > 15:
        score += 10
        reasons.append(f'overlap({overlap:.0f})')
    
    return {
        'score': score,
        'reason': ', '.join(reasons),
        'merge_type': merge_type,
        'distance': min_endpoint_dist
    }


def find_best_merge_candidates(whiskers, distance_threshold=30.0, min_score=0, fixture_points=None):
    """
    Find all good merge candidates and rank them by score.
    
    Returns:
    --------
    list : List of tuples (idx1, idx2, score_info) sorted by score (best first)
    """
    if fixture_points is None:
        fixture_points = []
    
    candidates = []
    
    for i, w1 in enumerate(whiskers):
        for j, w2 in enumerate(whiskers[i+1:], start=i+1):
            score_info = score_merge_candidate(w1, w2, distance_threshold, fixture_points)
            
            if score_info['score'] >= min_score:
                candidates.append((i, j, score_info))
    
    # Sort by score (highest first)
    candidates.sort(key=lambda x: x[2]['score'], reverse=True)
    
    return candidates


def merge_whiskers_by_type(whisker1, whisker2, merge_type):
    """
    Merge two whiskers based on connection type.
    
    Returns:
    --------
    tuple : (merged_x_coords, merged_y_coords)
    """
    x1 = whisker1['x_coords']
    y1 = whisker1['y_coords']
    x2 = whisker2['x_coords']
    y2 = whisker2['y_coords']
    
    if merge_type == 'tip1-base2':
        # Natural connection: tip1 to base2
        merged_x = x1 + x2
        merged_y = y1 + y2
    elif merge_type == 'base1-tip2':
        # Natural connection: tip2 to base1
        merged_x = x2 + x1
        merged_y = y2 + y1
    elif merge_type == 'base1-base2':
        # Reverse whisker1 and connect at bases
        merged_x = x1[::-1] + x2
        merged_y = y1[::-1] + y2
    elif merge_type == 'tip1-tip2':
        # Connect at tips
        merged_x = x1 + x2[::-1]
        merged_y = y1 + y2[::-1]
    else:
        # Default
        merged_x = x1 + x2
        merged_y = y1 + y2
    
    return merged_x, merged_y


def process_frame_smart(frame_df, target_whisker_count=5, 
                        distance_threshold=30.0,
                        min_merge_score=0,
                        verbose=False):
    """
    Smart processing: score all merge candidates and only merge the best ones.
    
    Returns:
    --------
    tuple : (processed_df, report)
    """
    frame_num = frame_df['Frame'].iloc[0]
    
    # If frame has target or fewer whiskers, return as-is
    if len(frame_df) <= target_whisker_count:
        return frame_df, {'merged': 0, 'original_count': len(frame_df), 'candidates_found': 0}
    
    # Parse all whiskers
    whiskers = []
    for idx, row in frame_df.iterrows():
        x_coords = parse_coordinates(row['X'])
        y_coords = parse_coordinates(row['Y'])
        
        if len(x_coords) < 2:
            continue
        
        base, tip = get_whisker_endpoints(x_coords, y_coords)
        angle = calculate_whisker_angle(x_coords, y_coords)
        length = calculate_whisker_length(x_coords, y_coords)
        
        whiskers.append({
            'index': idx,
            'x_coords': x_coords,
            'y_coords': y_coords,
            'base': base,
            'tip': tip,
            'angle': angle,
            'length': length,
            'row': row
        })
    
    # Identify fixture points (stable whisker bases)
    fixture_points = identify_fixture_points(whiskers, cluster_radius=25.0)
    
    # Filter out obvious noise whiskers
    non_noise = [w for w in whiskers if not is_noise_whisker(w, whiskers)]
    
    report = {
        'merged': 0,
        'original_count': len(whiskers),
        'noise_filtered': len(whiskers) - len(non_noise),
        'fixture_points_found': len(fixture_points),
        'candidates_found': 0,
        'merge_details': []
    }
    
    if verbose:
        print(f"  Identified {len(fixture_points)} fixture points (protected bases)")
    
    # Work with non-noise whiskers
    current_whiskers = non_noise.copy()
    merged_indices = set()
    
    # Keep merging until we reach target or no good candidates
    while len(current_whiskers) > target_whisker_count:
        # Find all merge candidates
        candidates = find_best_merge_candidates(
            current_whiskers,
            distance_threshold,
            min_score=min_merge_score,
            fixture_points=fixture_points
        )
        
        report['candidates_found'] = len(candidates)
        
        if not candidates:
            # No good merge candidates found
            break
        
        # Take the best candidate
        idx1, idx2, score_info = candidates[0]
        
        if verbose:
            print(f"  Frame {frame_num}: Merging whiskers {idx1} and {idx2}")
            print(f"    Score: {score_info['score']:.1f}, Reason: {score_info['reason']}")
            print(f"    Distance: {score_info['distance']:.1f}px, Type: {score_info['merge_type']}")
        
        w1 = current_whiskers[idx1]
        w2 = current_whiskers[idx2]
        
        # Merge them
        merged_x, merged_y = merge_whiskers_by_type(w1, w2, score_info['merge_type'])
        
        # Track merged indices
        if w1['index'] is not None:
            merged_indices.add(w1['index'])
        if w2['index'] is not None:
            merged_indices.add(w2['index'])
        
        # Create merged whisker
        merged_whisker = {
            'x_coords': merged_x,
            'y_coords': merged_y,
            'base': get_whisker_endpoints(merged_x, merged_y)[0],
            'tip': get_whisker_endpoints(merged_x, merged_y)[1],
            'angle': calculate_whisker_angle(merged_x, merged_y),
            'length': calculate_whisker_length(merged_x, merged_y),
            'index': None
        }
        
        # Remove merged whiskers and add new one
        new_whiskers = [w for i, w in enumerate(current_whiskers) if i not in [idx1, idx2]]
        new_whiskers.append(merged_whisker)
        
        current_whiskers = new_whiskers
        report['merged'] += 1
        report['merge_details'].append({
            'score': score_info['score'],
            'reason': score_info['reason'],
            'distance': score_info['distance'],
            'type': score_info['merge_type']
        })
    
    # Build result dataframe
    result_rows = []
    for whisker in current_whiskers:
        if whisker['index'] is None:
            # Merged whisker
            result_rows.append({
                'Frame': frame_num,
                'X': ','.join(map(str, whisker['x_coords'])),
                'Y': ','.join(map(str, whisker['y_coords']))
            })
        else:
            # Original whisker
            result_rows.append(whisker['row'].to_dict())
    
    result_df = pd.DataFrame(result_rows)
    report['final_count'] = len(result_df)
    
    return result_df, report


def smart_merge_dataset(csv_path, output_path=None,
                       target_whisker_count=5,
                       distance_threshold=30.0,
                       min_merge_score=0,
                       verbose=True):
    """
    Smart merge entire dataset.
    
    Parameters:
    -----------
    csv_path : str
        Input CSV path
    output_path : str
        Output CSV path (optional)
    target_whisker_count : int
        Target whiskers per frame
    distance_threshold : float
        Max distance for merging (liberal threshold)
    min_merge_score : int
        Minimum score to consider merge (higher = more conservative)
    verbose : bool
        Print details
    
    Returns:
    --------
    tuple : (result_df, statistics)
    """
    df = pd.read_csv(csv_path)
    
    if verbose:
        print(f"Loading: {csv_path}")
        print(f"Total rows: {len(df)}")
        print(f"Total frames: {df['Frame'].nunique()}")
        print(f"\nParameters:")
        print(f"  Distance threshold: {distance_threshold}px")
        print(f"  Min merge score: {min_merge_score}")
        print(f"  Target whisker count: {target_whisker_count}")
        print("="*60)
    
    stats = {
        'total_frames': 0,
        'frames_over_target': 0,
        'frames_processed': 0,
        'total_merges': 0,
        'frames_improved': 0,
        'noise_filtered': 0
    }
    
    result_frames = []
    frames = sorted(df['Frame'].unique())
    
    for frame_num in frames:
        frame_df = df[df['Frame'] == frame_num].copy()
        stats['total_frames'] += 1
        
        original_count = len(frame_df)
        
        if original_count > target_whisker_count:
            stats['frames_over_target'] += 1
            
            processed_df, report = process_frame_smart(
                frame_df,
                target_whisker_count,
                distance_threshold,
                min_merge_score,
                verbose=verbose
            )
            
            result_frames.append(processed_df)
            
            if report['merged'] > 0:
                stats['frames_processed'] += 1
                stats['total_merges'] += report['merged']
                stats['noise_filtered'] += report.get('noise_filtered', 0)
                
                if report['final_count'] <= target_whisker_count:
                    stats['frames_improved'] += 1
                
                if verbose:
                    print(f"Frame {frame_num}: {original_count} → {report['final_count']} whiskers")
                    print(f"  Merged: {report['merged']}, Noise filtered: {report.get('noise_filtered', 0)}")
                    if report['merge_details']:
                        for i, detail in enumerate(report['merge_details'], 1):
                            print(f"  Merge {i}: score={detail['score']:.0f}, dist={detail['distance']:.1f}px, {detail['reason']}")
                    print()
        else:
            result_frames.append(frame_df)
    
    result_df = pd.concat(result_frames, ignore_index=True)
    
    if output_path is None:
        base_name = os.path.splitext(csv_path)[0]
        output_path = f"{base_name}_smart_merged.csv"
    
    result_df.to_csv(output_path, index=False)
    
    if verbose:
        print(f"\n{'='*60}")
        print("SMART MERGING COMPLETE")
        print(f"{'='*60}")
        print(f"Total frames: {stats['total_frames']}")
        print(f"Frames with >{target_whisker_count} whiskers: {stats['frames_over_target']}")
        print(f"Frames where merging occurred: {stats['frames_processed']}")
        print(f"Total merges: {stats['total_merges']}")
        print(f"Noise whiskers filtered: {stats['noise_filtered']}")
        print(f"Frames improved to target: {stats['frames_improved']}")
        print(f"\nOriginal rows: {len(df)}")
        print(f"Final rows: {len(result_df)}")
        print(f"\nSaved to: {output_path}")
        print(f"{'='*60}")
    
    return result_df, stats
