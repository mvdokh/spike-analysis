import os
import cv2

VIDEO_EXTENSIONS = ['.mp4']

def find_videos(root_dir):
    video_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                video_files.append(os.path.join(dirpath, filename))
    return video_files

def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def main():
    # Set the default root directory to PCRt_TeLC, but allow override
    DEFAULT_ROOT = r'C:\Users\wanglab\Desktop\PCRt_TeLC'
    import argparse
    import csv

    parser = argparse.ArgumentParser(description='Count frames in all .mp4 videos recursively in a folder.')
    parser.add_argument('--root', type=str, default=DEFAULT_ROOT, help='Root directory to search for videos')
    parser.add_argument('--csv', type=str, default='video_frame_counts.csv', help='Output CSV file name')
    args = parser.parse_args()

    video_files = find_videos(args.root)
    if not video_files:
        print('No .mp4 videos found.')
        return
    print(f'Found {len(video_files)} video(s):\n')
    results = []
    for video in video_files:
        frames = count_frames(video)
        if frames is not None:
            print(f'{video}: {frames} frames')
            results.append({'video_path': video, 'frame_count': frames})
        else:
            print(f'{video}: Could not open video file')
            results.append({'video_path': video, 'frame_count': 'ERROR'})

    # Write results to CSV
    with open(args.csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['video_path', 'frame_count'])
        writer.writeheader()
        writer.writerows(results)
    print(f'\nResults written to {args.csv}')

if __name__ == '__main__':
    main()
