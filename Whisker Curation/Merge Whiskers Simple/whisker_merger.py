"""
Whisker Merger Script
Identifies frames with more than 5 whiskers and attempts to merge split whiskers.
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


def find_min_distance_between_whiskers(x1, y1, x2, y2):
    """
    Find the minimum distance between any two points on two whiskers.
    
    Returns:
    --------
    tuple : (min_distance, idx1, idx2)
        min_distance: the minimum distance found
        idx1: index of point on whisker 1
        idx2: index of point on whisker 2
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


def find_closest_whisker_pair(whiskers, distance_threshold=20.0, angle_threshold=np.pi/6):
    """
    Find the pair of whiskers that are closest at ANY point along their length.
    
    Parameters:
    -----------
    whiskers : list
        List of whisker dictionaries
    distance_threshold : float
        Maximum distance between any points to consider (pixels)
    angle_threshold : float
        Maximum angle difference to consider (radians)
        
    Returns:
    --------
    tuple : (idx1, idx2, closest_point1_idx, closest_point2_idx, distance) or (None, None, None, None, None)
    """
    best_pair = None
    best_distance = float('inf')
    
    for i, w1 in enumerate(whiskers):
        for j, w2 in enumerate(whiskers[i+1:], start=i+1):
            if len(w1['x_coords']) < 2 or len(w2['x_coords']) < 2:
                continue
            
            # Check angle similarity (optional constraint)
            if w1['angle'] is not None and w2['angle'] is not None:
                angle_diff = abs(w1['angle'] - w2['angle'])
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                
                if angle_diff > angle_threshold:
                    continue
            
            # Find minimum distance between ANY points on the two whiskers
            min_dist, pt1_idx, pt2_idx = find_min_distance_between_whiskers(
                w1['x_coords'], w1['y_coords'],
                w2['x_coords'], w2['y_coords']
            )
            
            # Only consider if within threshold and better than current best
            if min_dist <= distance_threshold and min_dist < best_distance:
                best_distance = min_dist
                best_pair = (i, j, pt1_idx, pt2_idx, min_dist)
    
    if best_pair is None:
        return None, None, None, None, None
    
    return best_pair


def merge_whiskers(whisker1, whisker2, pt1_idx, pt2_idx):
    """
    Merge two whiskers by connecting them at the specified closest points.
    Creates a path from one end to the connection points, then to the other end.
    
    Parameters:
    -----------
    whisker1, whisker2 : dict
        Whisker dictionaries with x_coords and y_coords
    pt1_idx : int
        Index of closest point on whisker1
    pt2_idx : int
        Index of closest point on whisker2
    
    Returns:
    --------
    tuple : (merged_x_coords, merged_y_coords)
    """
    x1 = whisker1['x_coords']
    y1 = whisker1['y_coords']
    x2 = whisker2['x_coords']
    y2 = whisker2['y_coords']
    
    # Strategy: connect all points from both whiskers through the closest connection
    # Take whisker1 up to pt1_idx, then whisker2 from pt2_idx onwards, then remaining parts
    
    # Simple approach: concatenate whisker1 up to connection, then whisker2 from connection
    merged_x = x1[:pt1_idx+1] + x2[pt2_idx:]
    merged_y = y1[:pt1_idx+1] + y2[pt2_idx:]
    
    # Add remaining part of whisker1 (after connection point)
    if pt1_idx < len(x1) - 1:
        merged_x.extend(x1[pt1_idx+1:])
        merged_y.extend(y1[pt1_idx+1:])
    
    # Add remaining part of whisker2 (before connection point) in reverse
    if pt2_idx > 0:
        merged_x.extend(x2[:pt2_idx][::-1])
        merged_y.extend(y2[:pt2_idx][::-1])
    
    return merged_x, merged_y


def process_frame(frame_df, target_whisker_count=5, 
                  distance_threshold=20.0, 
                  angle_threshold=np.pi/6):
    """
    Process a single frame: if it has more than target whiskers, try to merge the closest pair.
    
    Parameters:
    -----------
    frame_df : pd.DataFrame
        DataFrame with whiskers from one frame
    target_whisker_count : int
        Target number of whiskers (default: 5)
    distance_threshold : float
        Maximum distance for merging (pixels)
    angle_threshold : float
        Maximum angle difference for merging (radians)
        
    Returns:
    --------
    tuple : (processed_df, merge_report)
    """
    frame_num = frame_df['Frame'].iloc[0]
    
    # If frame has target or fewer whiskers, return as-is
    if len(frame_df) <= target_whisker_count:
        return frame_df, {'merged': 0, 'original_count': len(frame_df)}
    
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
    
    merge_report = {'merged': 0, 'original_count': len(whiskers)}
    merged_indices = set()
    result_rows = []
    
    # Keep merging until we reach target or can't merge anymore
    current_whiskers = whiskers.copy()
    
    while len(current_whiskers) > target_whisker_count:
        # Find best pair to merge
        idx1, idx2, pt1_idx, pt2_idx, _ = find_closest_whisker_pair(
            current_whiskers, 
            distance_threshold, 
            angle_threshold
        )
        
        if idx1 is None:
            # No suitable pairs found, stop merging
            break
        
        # Merge the pair
        w1 = current_whiskers[idx1]
        w2 = current_whiskers[idx2]
        
        merged_x, merged_y = merge_whiskers(w1, w2, pt1_idx, pt2_idx)
        
        # Mark original indices as merged
        merged_indices.add(w1['index'])
        merged_indices.add(w2['index'])
        
        # Create merged whisker entry
        merged_whisker = {
            'x_coords': merged_x,
            'y_coords': merged_y,
            'base': get_whisker_endpoints(merged_x, merged_y)[0],
            'tip': get_whisker_endpoints(merged_x, merged_y)[1],
            'angle': calculate_whisker_angle(merged_x, merged_y),
            'length': calculate_whisker_length(merged_x, merged_y),
            'index': None  # New merged whisker
        }
        
        # Remove merged whiskers and add the new one
        new_whiskers = []
        for i, w in enumerate(current_whiskers):
            if i not in [idx1, idx2]:
                new_whiskers.append(w)
        new_whiskers.append(merged_whisker)
        
        current_whiskers = new_whiskers
        merge_report['merged'] += 1
    
    # Build result dataframe
    for whisker in current_whiskers:
        if whisker['index'] is None:
            # This is a merged whisker - create dict
            result_rows.append({
                'Frame': frame_num,
                'X': ','.join(map(str, whisker['x_coords'])),
                'Y': ','.join(map(str, whisker['y_coords']))
            })
        else:
            # Original whisker that wasn't merged - convert Series to dict
            result_rows.append(whisker['row'].to_dict())
    
    result_df = pd.DataFrame(result_rows)
    merge_report['final_count'] = len(result_df)
    
    return result_df, merge_report


def merge_whiskers_in_dataset(csv_path, output_path=None,
                               target_whisker_count=5,
                               distance_threshold=20.0,
                               angle_threshold=np.pi/6,
                               verbose=True):
    """
    Process entire dataset, merging whiskers in frames with more than target count.
    
    Parameters:
    -----------
    csv_path : str
        Path to input CSV file
    output_path : str
        Path to save output CSV (if None, generates automatic name)
    target_whisker_count : int
        Target number of whiskers per frame
    distance_threshold : float
        Maximum distance for merging (pixels)
    angle_threshold : float
        Maximum angle difference for merging (radians)
    verbose : bool
        Print progress information
        
    Returns:
    --------
    tuple : (result_df, statistics)
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    if verbose:
        print(f"Loading: {csv_path}")
        print(f"Total rows: {len(df)}")
        print(f"Total frames: {df['Frame'].nunique()}")
    
    # Statistics
    stats = {
        'total_frames': 0,
        'frames_over_target': 0,
        'frames_processed': 0,
        'total_merges': 0,
        'frames_improved': 0
    }
    
    # Process each frame
    result_frames = []
    frames = sorted(df['Frame'].unique())
    
    for frame_num in frames:
        frame_df = df[df['Frame'] == frame_num].copy()
        stats['total_frames'] += 1
        
        original_count = len(frame_df)
        
        if original_count > target_whisker_count:
            stats['frames_over_target'] += 1
            
            # Process this frame
            processed_df, merge_report = process_frame(
                frame_df,
                target_whisker_count,
                distance_threshold,
                angle_threshold
            )
            
            result_frames.append(processed_df)
            
            if merge_report['merged'] > 0:
                stats['frames_processed'] += 1
                stats['total_merges'] += merge_report['merged']
                
                if merge_report['final_count'] <= target_whisker_count:
                    stats['frames_improved'] += 1
                
                if verbose:
                    print(f"Frame {frame_num}: {original_count} → {merge_report['final_count']} whiskers ({merge_report['merged']} merges)")
        else:
            result_frames.append(frame_df)
    
    # Combine all frames
    result_df = pd.concat(result_frames, ignore_index=True)
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(csv_path)[0]
        output_path = f"{base_name}_merged.csv"
    
    # Save
    result_df.to_csv(output_path, index=False)
    
    if verbose:
        print(f"\n{'='*60}")
        print("MERGING COMPLETE")
        print(f"{'='*60}")
        print(f"Total frames: {stats['total_frames']}")
        print(f"Frames with >{target_whisker_count} whiskers: {stats['frames_over_target']}")
        print(f"Frames where merging occurred: {stats['frames_processed']}")
        print(f"Total whisker pairs merged: {stats['total_merges']}")
        print(f"Frames improved to target: {stats['frames_improved']}")
        print(f"\nOriginal rows: {len(df)}")
        print(f"Final rows: {len(result_df)}")
        print(f"\nSaved to: {output_path}")
        print(f"{'='*60}")
    
    return result_df, stats


if __name__ == "__main__":
    csv_path = "lines_output.csv"
    
    if os.path.exists(csv_path):
        result_df, stats = merge_whiskers_in_dataset(
            csv_path,
            target_whisker_count=5,
            distance_threshold=20.0,
            angle_threshold=np.pi/6,
            verbose=True
        )
    else:
        print(f"Error: File '{csv_path}' not found.")
