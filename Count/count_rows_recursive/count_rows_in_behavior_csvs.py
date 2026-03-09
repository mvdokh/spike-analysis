import os
import csv

def find_behavior_csvs(root_dir):
    csv_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('behavior.csv'):
                csv_files.append(os.path.join(dirpath, filename))
    return csv_files

def find_video_in_folder(folder):
    # Find the first .mp4 file in the folder
    for f in os.listdir(folder):
        if f.lower().endswith('.mp4'):
            return os.path.join(folder, f)
    return None

def count_frames(video_path):
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count
    except Exception as e:
        return None

def count_rows(csv_path):
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f) - 1  # subtract 1 for header
    except Exception as e:
        return f'ERROR: {e}'

def main():
    import argparse
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import numpy as np
    import re

    parser = argparse.ArgumentParser(description='Count rows in all *behavior.csv files recursively in a folder.')
    parser.add_argument('--root', type=str, default=r'C:\Users\wanglab\Desktop\IRt_Bipoles_1-20-26\TeLC', help='Root directory to search for behavior.csv files')
    parser.add_argument('--csv', type=str, default='behavior_csv_row_counts.csv', help='Output CSV file name (will be saved in script folder)')
    args = parser.parse_args()

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_output_path = os.path.join(script_dir, args.csv)
    svg_output_path = os.path.join(script_dir, 'behavior_csv_row_counts_bar_chart.svg')

    csv_files = find_behavior_csvs(args.root)
    if not csv_files:
        print('No behavior.csv files found.')
        return
    print(f'Found {len(csv_files)} behavior.csv file(s):\n')
    results = []
    for csv_file in csv_files:
        row_count = count_rows(csv_file)
        folder = os.path.dirname(csv_file)
        video_path = find_video_in_folder(folder)
        if video_path:
            frame_count = count_frames(video_path)
        else:
            frame_count = None
        # Normalize if possible
        try:
            norm = float(row_count) / float(frame_count) if frame_count and frame_count > 0 else None
        except Exception:
            norm = None
        print(f'{csv_file}: {row_count} rows, {frame_count if frame_count is not None else "NO VIDEO"} frames, normalized: {norm if norm is not None else "N/A"}')
        results.append({'csv_path': csv_file, 'row_count': row_count, 'video_path': video_path if video_path else '', 'frame_count': frame_count if frame_count is not None else '', 'normalized': norm if norm is not None else ''})

    # Prepare data for bar chart: folder name (parent of csv) and normalized value
    folder_norms = defaultdict(list)
    for r in results:
        folder = os.path.basename(os.path.dirname(r['csv_path']))
        # Get the grandparent folder (e.g., Phox2b#42, Phox2b#39, etc.)
        parent = os.path.basename(os.path.dirname(os.path.dirname(r['csv_path'])))
        # Extract the #XX from the parent folder name
        match = re.search(r'#\d+', parent)
        prefix = match.group(0) if match else parent
        label = f'{prefix}_{folder}'
        try:
            norm = float(r['normalized']) if r['normalized'] != '' else None
        except Exception:
            norm = None
        if norm is not None:
            folder_norms[label].append(norm)

    # Use the mean normalized value per folder label
    folders = sorted(folder_norms.keys())
    norms = [np.mean(folder_norms[f]) for f in folders]

    plt.figure(figsize=(max(8, len(folders)*0.8), 6))
    bar_colors = ['red' if f.endswith('_1') else 'blue' for f in folders]
    plt.bar(folders, norms, color=bar_colors)
    plt.xlabel('Folder (Parent_Number_Folder)')
    plt.ylabel('Normalized Rows (rows/frames)')
    plt.title('Normalized Rows in behavior.csv per Folder')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(svg_output_path)

    with open(csv_output_path, 'w', newline='', encoding='utf-8') as out_csv:
        writer = csv.DictWriter(out_csv, fieldnames=['csv_path', 'row_count', 'video_path', 'frame_count', 'normalized'])
        writer.writeheader()
        writer.writerows(results)
    print(f'\nResults written to {csv_output_path}')

    print(f'Bar chart saved as {svg_output_path}')

    # --- Compute difference in number of rows between bottom and side for each (#XX_YYYY) group ---
    # Map: group_key (e.g., #38_0307) -> {'bottom': value, 'side': value}
    group_row_counts = {}
    for r in results:
        folder = os.path.basename(os.path.dirname(r['csv_path']))
        parent = os.path.basename(os.path.dirname(os.path.dirname(r['csv_path'])))
        match = re.search(r'#\d+', parent)
        prefix = match.group(0) if match else parent
        label = f'{prefix}_{folder}'
        base_label = label[:-2] if label.endswith('_1') else label
        if base_label not in group_row_counts:
            group_row_counts[base_label] = {'bottom': None, 'side': None}
        try:
            row_count = int(r['row_count']) if r['row_count'] != '' and not isinstance(r['row_count'], str) or r['row_count'].isdigit() else None
        except Exception:
            row_count = None
        if label.endswith('_1'):
            group_row_counts[base_label]['side'] = row_count
        else:
            group_row_counts[base_label]['bottom'] = row_count

    # Prepare data for difference plot (rows)
    diff_labels = []
    diffs = []
    for group_key, vals in sorted(group_row_counts.items()):
        if vals['bottom'] is not None and vals['side'] is not None:
            diff = vals['side'] - vals['bottom']
            diff_labels.append(group_key)
            diffs.append(diff)

    # Plot the difference (rows)
    if diff_labels:
        plt.figure(figsize=(max(8, len(diff_labels)*0.8), 6))
        bar_colors = ['purple' if d >= 0 else 'green' for d in diffs]
        plt.bar(diff_labels, diffs, color=bar_colors)
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.xlabel('Folder (#XX_YYYY)')
        plt.ylabel('Side - Bottom (rows)')
        plt.title('Difference between bottom and side (rows)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        diff_svg_path = os.path.join(script_dir, 'behavior_csv_row_counts_difference_bar_chart.svg')
        plt.savefig(diff_svg_path)
        print(f'Difference plot saved as {diff_svg_path}')
    import argparse
    import matplotlib.pyplot as plt
    from collections import defaultdict
    parser = argparse.ArgumentParser(description='Count rows in all *behavior.csv files recursively in a folder.')
    parser.add_argument('--root', type=str, default=r'C:\Users\wanglab\Desktop\IRt_Bipoles_1-20-26\TeLC', help='Root directory to search for behavior.csv files')
    parser.add_argument('--csv', type=str, default='behavior_csv_row_counts.csv', help='Output CSV file name (will be saved in script folder)')
    args = parser.parse_args()

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_output_path = os.path.join(script_dir, args.csv)
    svg_output_path = os.path.join(script_dir, 'behavior_csv_row_counts_bar_chart.svg')

    csv_files = find_behavior_csvs(args.root)
    if not csv_files:
        print('No behavior.csv files found.')
        return
    print(f'Found {len(csv_files)} behavior.csv file(s):\n')
    results = []
    for csv_file in csv_files:
        row_count = count_rows(csv_file)
        folder = os.path.dirname(csv_file)
        video_path = find_video_in_folder(folder)
        if video_path:
            frame_count = count_frames(video_path)
        else:
            frame_count = None
        # Normalize if possible
        try:
            norm = float(row_count) / float(frame_count) if frame_count and frame_count > 0 else None
        except Exception:
            norm = None
        print(f'{csv_file}: {row_count} rows, {frame_count if frame_count is not None else "NO VIDEO"} frames, normalized: {norm if norm is not None else "N/A"}')
        results.append({'csv_path': csv_file, 'row_count': row_count, 'video_path': video_path if video_path else '', 'frame_count': frame_count if frame_count is not None else '', 'normalized': norm if norm is not None else ''})

    # Prepare data for bar chart: folder name (parent of csv) and normalized value
    folder_norms = defaultdict(list)
    for r in results:
        folder = os.path.basename(os.path.dirname(r['csv_path']))
        # Get the grandparent folder (e.g., Phox2b#42, Phox2b#39, etc.)
        parent = os.path.basename(os.path.dirname(os.path.dirname(r['csv_path'])))
        # Extract the #XX from the parent folder name
        import re
        match = re.search(r'#\d+', parent)
        prefix = match.group(0) if match else parent
        label = f'{prefix}_{folder}'
        try:
            norm = float(r['normalized']) if r['normalized'] != '' else None
        except Exception:
            norm = None
        if norm is not None:
            folder_norms[label].append(norm)

    # Use the mean normalized value per folder label
    import numpy as np
    folders = sorted(folder_norms.keys())
    norms = [np.mean(folder_norms[f]) for f in folders]

    plt.figure(figsize=(max(8, len(folders)*0.8), 6))
    bar_colors = ['red' if f.endswith('_1') else 'blue' for f in folders]
    plt.bar(folders, norms, color=bar_colors)
    plt.xlabel('Folder (Parent_Number_Folder)')
    plt.ylabel('Normalized Rows (rows/frames)')
    plt.title('Normalized Rows in behavior.csv per Folder')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(svg_output_path)

    with open(csv_output_path, 'w', newline='', encoding='utf-8') as out_csv:
        writer = csv.DictWriter(out_csv, fieldnames=['csv_path', 'row_count', 'video_path', 'frame_count', 'normalized'])
        writer.writeheader()
        writer.writerows(results)
    print(f'\nResults written to {csv_output_path}')

    print(f'Bar chart saved as {svg_output_path}')

if __name__ == '__main__':
    main()
