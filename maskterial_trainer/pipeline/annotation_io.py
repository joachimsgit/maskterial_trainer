from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def mask_path_for_image(project_path: Path, image_path: Path) -> Path:
    return project_path / "instance_masks" / f"{image_path.stem}.png"


def load_instance_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.array(Image.open(path))
    return arr.astype(np.int32)


def save_instance_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = mask.astype(np.uint16)
    Image.fromarray(arr).save(path)
