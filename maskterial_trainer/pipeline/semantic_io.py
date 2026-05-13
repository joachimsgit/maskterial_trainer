from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def semantic_mask_path_for_image(project_path: Path, image_path: Path) -> Path:
    return project_path / "semantic_masks" / f"{image_path.stem}.png"


def load_semantic_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.array(Image.open(path)).astype(np.uint8)


def save_semantic_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(path)
