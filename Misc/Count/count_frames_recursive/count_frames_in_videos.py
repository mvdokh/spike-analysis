import csv
import os

import cv2

ROOT_DIR = r"C:\Users\wanglab\Desktop\Ina\PCRt_TeLC"
OUTPUT_CSV = os.path.join(ROOT_DIR, "video_frame_counts.csv")
VIDEO_EXTENSIONS = (".mp4",)


def find_videos(root_dir):
    videos = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(VIDEO_EXTENSIONS):
                videos.append(os.path.join(dirpath, filename))
    return sorted(videos)


def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def get_pre_post(video_path):
    parent = os.path.basename(os.path.dirname(video_path)).lower()
    if parent.endswith("_pre"):
        return "pre"
    if parent.endswith("_post"):
        return "post"
    return "unknown"


def main():
    video_files = find_videos(ROOT_DIR)
    if not video_files:
        print(f"No videos found in {ROOT_DIR}")
        return

    results = []
    for video_path in video_files:
        video_name = os.path.basename(video_path)
        pre_post = get_pre_post(video_path)
        frames = count_frames(video_path)
        if frames is None:
            print(f"{video_name} ({pre_post}): could not open")
            results.append((video_name, pre_post, "ERROR"))
        else:
            print(f"{video_name} ({pre_post}): {frames}")
            results.append((video_name, pre_post, frames))

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["video_name", "pre_post", "frame_count"])
        writer.writerows(results)

    print(f"\nWrote {len(results)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
