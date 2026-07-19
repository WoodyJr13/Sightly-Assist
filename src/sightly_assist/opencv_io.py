"""Optional OpenCV-backed replay image loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sightly_assist.perception import FramePacket


class VisionDependencyError(RuntimeError):
    """Raised when an optional computer-vision dependency is unavailable."""


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise VisionDependencyError(
            'OpenCV is not installed. Install Sightly Assist with the "vision" extra.'
        ) from exc
    return cv2


def load_rgb(frame: FramePacket, root: Path) -> np.ndarray:
    """Load one RGB frame and verify that its dimensions match the manifest."""

    cv2 = _import_cv2()
    path = root / frame.rgb_path
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not decode RGB frame: {path}")

    height, width = image.shape[:2]
    if width != frame.width_px or height != frame.height_px:
        raise ValueError(
            "Decoded RGB dimensions do not match manifest: "
            f"expected {frame.width_px}x{frame.height_px}, got {width}x{height}"
        )
    return image


def load_depth(frame: FramePacket, root: Path) -> np.ndarray | None:
    """Load a NumPy depth map when the replay frame references one."""

    if frame.depth_path is None:
        return None

    path = root / frame.depth_path
    try:
        depth = np.load(path, allow_pickle=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Depth frame does not exist: {path}") from None

    if depth.ndim != 2:
        raise ValueError(f"Depth frame must be two-dimensional: {path}")
    if depth.shape != (frame.height_px, frame.width_px):
        raise ValueError(
            "Depth dimensions do not match manifest: "
            f"expected {(frame.height_px, frame.width_px)}, got {depth.shape}"
        )
    if not np.issubdtype(depth.dtype, np.number):
        raise ValueError(f"Depth frame must use a numeric data type: {path}")
    return depth
