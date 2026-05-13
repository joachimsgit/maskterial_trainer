"""Flatfield (vignette) correction.

Formula mirrors MaskTerial_Repo/tools/utils/preprocessor_functions.remove_vignette:

    corrected = image / flatfield * mean(flatfield), clipped to <= max_background

Used at COCO Conversion time to stage corrected copies of every referenced
image into `<project>/coco/images/`. Raw images at the project root are
never modified.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MAX_BACKGROUND_VALUE = 241


def load_flatfield(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load flatfield image: {path}")
    return img


def remove_vignette(
    image: np.ndarray,
    flatfield: np.ndarray,
    max_background: int = MAX_BACKGROUND_VALUE,
) -> np.ndarray:
    """Returns the corrected image (uint8 BGR)."""
    if flatfield.shape != image.shape:
        raise ValueError(
            f"Flatfield shape {flatfield.shape} != image shape {image.shape}"
        )
    flatfield_mean = np.array(cv2.mean(flatfield)[:3], dtype=np.float32)
    flatfield_f = flatfield.astype(np.float32)
    flatfield_f = np.where(flatfield_f < 1.0, 1.0, flatfield_f)
    corrected = image.astype(np.float32) / flatfield_f * flatfield_mean
    corrected[corrected > max_background] = max_background
    return corrected.clip(0, 255).astype(np.uint8)
