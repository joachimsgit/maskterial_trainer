from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .annotation_io import load_instance_mask, mask_path_for_image
from .flatfield import load_flatfield, remove_vignette
from .project import MaterialProject

MIN_INSTANCE_PIXELS_AFTER_ERODE = 200
ERODE_ITERATIONS = 3


@dataclass
class InstanceContrast:
    image_name: str
    label: int
    pixel_count: int
    mean_bgr: np.ndarray  # shape (3,), order B G R
    std_bgr: np.ndarray  # shape (3,), order B G R


def calculate_background_color(
    image_bgr: np.ndarray, radius: int = 10
) -> np.ndarray:
    """Mode-of-histogram background, matching MaskTerial's data_loader."""
    masks = []
    for i in range(3):
        ch = image_bgr[:, :, i]
        valid = cv2.inRange(ch, 20, 230)
        hist = cv2.calcHist([ch], [0], valid, [256], [0, 256])
        mode = int(np.argmax(hist))
        thresholded = cv2.inRange(
            ch, max(0, mode - radius), min(255, mode + radius)
        )
        eroded = cv2.erode(thresholded, np.ones((3, 3), np.uint8), iterations=3)
        masks.append(eroded)
    final = cv2.bitwise_and(masks[0], masks[1])
    final = cv2.bitwise_and(final, masks[2])
    if cv2.countNonZero(final) == 0:
        return np.zeros(3, dtype=np.float64)
    return np.array(cv2.mean(image_bgr, mask=final)[:3])


def compute_contrast_image(image_bgr: np.ndarray) -> np.ndarray | None:
    bg = calculate_background_color(image_bgr)
    if np.any(bg < 1):
        return None
    return image_bgr.astype(np.float32) / bg.astype(np.float32) - 1.0


def compute_instance_contrasts(
    image_bgr: np.ndarray,
    instance_mask: np.ndarray,
    image_name: str = "",
) -> list[InstanceContrast]:
    contrast = compute_contrast_image(image_bgr)
    if contrast is None:
        return []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out: list[InstanceContrast] = []
    for label in np.unique(instance_mask):
        if label == 0:
            continue
        mask = (instance_mask == label).astype(np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=ERODE_ITERATIONS)
        if np.count_nonzero(eroded) < MIN_INSTANCE_PIXELS_AFTER_ERODE:
            eroded = mask
        if np.count_nonzero(eroded) == 0:
            continue
        pixels = contrast[eroded != 0]
        out.append(
            InstanceContrast(
                image_name=image_name,
                label=int(label),
                pixel_count=int(np.count_nonzero(eroded)),
                mean_bgr=pixels.mean(axis=0),
                std_bgr=pixels.std(axis=0),
            )
        )
    return out


def aggregate_project_contrasts(
    project: MaterialProject,
    progress: callable | None = None,
) -> list[InstanceContrast]:
    images = project.list_raw_images()
    flatfield_img = None
    if (
        project.flatfield_enabled
        and project.flatfield_full_path is not None
        and project.flatfield_full_path.exists()
    ):
        try:
            flatfield_img = load_flatfield(project.flatfield_full_path)
        except Exception:
            flatfield_img = None

    out: list[InstanceContrast] = []
    for i, image_path in enumerate(images):
        if progress is not None:
            progress(i, len(images), image_path.name)
        mask_path = mask_path_for_image(project.path, image_path)
        if not mask_path.exists():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        if flatfield_img is not None and image.shape == flatfield_img.shape:
            image = remove_vignette(image, flatfield_img)
        mask = load_instance_mask(mask_path)
        if mask is None or mask.shape != image.shape[:2]:
            continue
        out.extend(compute_instance_contrasts(image, mask, image_path.name))
    if progress is not None:
        progress(len(images), len(images), "")
    return out
