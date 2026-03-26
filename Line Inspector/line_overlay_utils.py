from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np


FRAME_RE = re.compile(r"(\d+)")


def parse_frame_number(path_like: str | Path) -> int:
    """
    Extract the last integer group from a filename like 'img0048505.png' or '0048505.csv'.
    """
    s = Path(path_like).name
    matches = FRAME_RE.findall(s)
    if not matches:
        raise ValueError(f"Could not parse frame number from: {s!r}")
    return int(matches[-1])


def frame_to_id(frame: int, pad: int = 7) -> str:
    return f"{int(frame):0{pad}d}"


@dataclass(frozen=True)
class DatasetPaths:
    dataset_dir: Path
    images_dirname: str = "images"
    labels_dirname: str = "labels"
    image_prefix: str = "img"
    image_ext: str = ".png"
    frame_pad: int = 7

    @property
    def images_dir(self) -> Path:
        return self.dataset_dir / self.images_dirname

    @property
    def labels_dir(self) -> Path:
        return self.dataset_dir / self.labels_dirname

    def image_path(self, frame: int) -> Path:
        fid = frame_to_id(frame, self.frame_pad)
        return self.images_dir / f"{self.image_prefix}{fid}{self.image_ext}"

    def label_csv_path(self, label_type: str | int, frame: int) -> Path:
        fid = frame_to_id(frame, self.frame_pad)
        return self.labels_dir / str(label_type) / f"{fid}.csv"


def list_label_types(labels_dir: str | Path) -> list[str]:
    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        return []
    return sorted([p.name for p in labels_dir.iterdir() if p.is_dir()], key=lambda x: (len(x), x))


def list_frames_in_images_dir(
    images_dir: str | Path,
    image_prefix: str = "img",
    image_ext: str = ".png",
) -> list[int]:
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return []
    frames: list[int] = []
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        if image_ext and p.suffix.lower() != image_ext.lower():
            continue
        if image_prefix and not p.stem.startswith(image_prefix):
            continue
        try:
            frames.append(parse_frame_number(p.name))
        except ValueError:
            pass
    return sorted(set(frames))


def load_polyline_csv(csv_path: str | Path) -> np.ndarray:
    """
    Load a headerless CSV with 2 columns: x,y per row.
    Returns an array with shape (N, 2). Empty if file missing or no points.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return np.zeros((0, 2), dtype=float)

    try:
        pts = np.loadtxt(csv_path, delimiter=",", dtype=float)
    except ValueError:
        # Sometimes a single point loads as shape (2,)
        pts = np.loadtxt(csv_path, delimiter=",", dtype=float, ndmin=2)

    if pts.size == 0:
        return np.zeros((0, 2), dtype=float)

    pts = np.asarray(pts, dtype=float)
    if pts.ndim == 1:
        if pts.shape[0] != 2:
            raise ValueError(f"Expected 2 columns in {csv_path}, got shape {pts.shape}")
        pts = pts.reshape(1, 2)
    if pts.shape[1] != 2:
        raise ValueError(f"Expected 2 columns in {csv_path}, got shape {pts.shape}")
    return pts


def render_overlay_matplotlib(
    image_path: str | Path,
    polyline_xy: np.ndarray,
    *,
    alpha: float = 0.5,
    line_color: str = "red",
    line_width: float = 2.0,
    point_size: float = 10.0,
    show_points: bool = False,
    ax=None,
):
    """
    Render image + connected polyline overlay. If ax is None, creates a new figure+axes.
    """
    from PIL import Image
    import matplotlib.pyplot as plt

    image_path = Path(image_path)
    img = Image.open(image_path)

    created = False
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
        created = True

    ax.imshow(img)
    if polyline_xy is not None and len(polyline_xy) > 0:
        x = polyline_xy[:, 0]
        y = polyline_xy[:, 1]
        ax.plot(x, y, color=line_color, linewidth=line_width, alpha=alpha)
        if show_points:
            ax.scatter(x, y, s=point_size, color=line_color, alpha=alpha)

    ax.set_title(image_path.name)
    ax.set_axis_off()

    if created:
        plt.tight_layout()
        plt.show()
    return ax

