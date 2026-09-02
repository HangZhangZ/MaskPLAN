"""Shared, lightweight utilities for MaskPLAN inference scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from shapely.geometry import Polygon


REPO_ROOT = Path(__file__).resolve().parents[1]


def sample_partial_indices(valid_rooms: int, ratio: float, rng=np.random) -> np.ndarray:
    """Sample one-based room-token indices for a partial-input attribute."""
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("partial-input ratios must be between 0 and 1")
    if valid_rooms <= 0:
        return np.empty(0, dtype=np.int64)
    count = round(valid_rooms * ratio)
    return np.asarray(rng.choice(valid_rooms, count, replace=False) + 1, dtype=np.int64)


def partial_ratio(value: str) -> float:
    """Argparse converter for partial-input ratios."""
    ratio = float(value)
    if not 0.0 <= ratio <= 1.0:
        raise argparse.ArgumentTypeError("partial-input ratios must be between 0 and 1")
    return ratio


def set_random_seed(seed: Optional[int], tensorflow_module) -> None:
    """Seed MaskPLAN's NumPy and TensorFlow random sources when requested."""
    if seed is None:
        return
    np.random.seed(seed)
    tensorflow_module.random.set_seed(seed)


def largest_polygon(geometry) -> Optional[Polygon]:
    """Return the largest non-empty polygonal component, or ``None``."""
    if geometry is None or geometry.is_empty:
        return None
    if geometry.geom_type == "Polygon":
        return geometry if geometry.area > 0 else None
    polygons = []
    if hasattr(geometry, "geoms"):
        for component in geometry.geoms:
            polygon = largest_polygon(component)
            if polygon is not None:
                polygons.append(polygon)
    return max(polygons, key=lambda item: item.area) if polygons else None


def largest_contour(contours):
    """Return the largest non-degenerate OpenCV contour, or ``None``."""
    valid = [contour for contour in contours if len(contour) >= 3 and cv2.contourArea(contour) > 0]
    return max(valid, key=cv2.contourArea) if valid else None


def load_boundary_points(path: Path) -> np.ndarray:
    """Load and vectorize an inference boundary image with a useful error."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(
            "Boundary image not found at %s. Extract parsed_img/img_room_sqe/0.7z "
            "so that parsed_img/img_room_sqe/0/<site_id>.png exists, or run dataset "
            "preparation step (a)." % path
        )
    channel = image[:, :, -1].copy() if image.ndim == 3 else image.copy()
    channel[channel > 100] = 255
    _, threshold = cv2.threshold(channel, 120, 255, 0)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([[0, 0], [0, 127], [127, 127], [127, 0]], dtype=np.int32)
    contour = max(contours, key=cv2.contourArea)
    points = np.squeeze(cv2.approxPolyDP(contour, channel.shape[0] / 128, True))
    if points.ndim != 2 or len(points) < 3:
        return np.array([[0, 0], [0, 127], [127, 127], [127, 0]], dtype=np.int32)
    return np.asarray(points, dtype=np.int32)
