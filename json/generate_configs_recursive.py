import os
import json
import argparse
from glob import glob

parent_folder = "C:\\Users\\wanglab\\Desktop\\PCRt_TeLC\\Phox2b#38"

# Template for the config.json structure
TEMPLATE = [
    {
        "filepath": None,  # to be filled
        "data_type": "video",
        "name": "media"
    },
    {
        "filepath": None,  # to be filled
        "data_type": "mask",
        "name": "Tongue",
        "frame_key": "frames",
        "format": "hdf5",
        "probability_key": "probs",
        "height": 256,
        "width": 256,
        "y_key": "heights",
        "x_key": "widths",
        "color": "#FF0000",
        "operations": [
            {"type": "area"}
        ]
    },
    {
        "filepath": None,  # to be filled
        "data_type": "points",
        "name": "Jaw",
        "format": "csv",
        "frame_column": 0,
        "x_column": 1,
        "y_column": 2,
        "color": "#00FF00",
        "height": 256,
        "width": 256,
        "delim": " ",
        "scaled_height": 480,
        "scaled_width": 640
    }
]

# File patterns for each data type
PATTERNS = {
    'video': ['*.mp4', '*.avi', '*.mov'],
    'mask': ['*tongue.h5', '*.h5'],
    'points': ['*jaw.csv', '*.csv']
}

def find_file(subfolder, patterns):
    for pattern in patterns:
        matches = glob(os.path.join(subfolder, pattern))
        if matches:
            return os.path.basename(matches[0])
    return None

def create_config(subfolder):
    config = json.loads(json.dumps(TEMPLATE))  # deep copy
    config[0]['filepath'] = find_file(subfolder, PATTERNS['video'])
    config[1]['filepath'] = find_file(subfolder, PATTERNS['mask'])
    config[2]['filepath'] = find_file(subfolder, PATTERNS['points'])
    return config

def main(parent_folder):
    for root, dirs, files in os.walk(parent_folder):
        # Only create config if at least one data file is present
        if any(find_file(root, PATTERNS[k]) for k in PATTERNS):
            config = create_config(root)
            config_path = os.path.join(root, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Wrote config.json to {root}")

# Set your parent folder here:
PARENT_FOLDER = r"C:\Users\wanglab\Desktop\PCRt_TeLC\Phox2b#42"

if __name__ == "__main__":
    main(PARENT_FOLDER)
