"""Build the training-zip the inference server's /train/m2f endpoint expects.

Server-side flow (`MaskTerial_Repo/server.py`):
    extract zip into data_dir/
    read data_dir/result.json
    convert polygon segmentations to RLE  (RLE entries are passed through)
    strip leading "images/" from file_name
    train

So the zip we produce must contain:
    result.json     COCO with file_name = "images/<name>"
    images/<name>   for every image referenced

Our project's coco/train_annotations.json already has RLE segmentations — the
server's converter passes those through unchanged, so we don't need any format
conversion. We only need to prefix file_name with "images/" and bundle the
files.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Callable

from .project import MaterialProject
from .training import training_image_root

ProgressCallback = Callable[[str], None]


def build_training_zip(
    project: MaterialProject,
    dest: Path,
    progress: ProgressCallback | None = None,
) -> dict:
    coco_path = project.path / "coco" / "train_annotations.json"
    if not coco_path.exists():
        raise FileNotFoundError(
            "train_annotations.json not found — run COCO Conversion first."
        )

    if progress:
        progress("Reading COCO file…")
    coco_data = json.loads(coco_path.read_text())

    n_anns = len(coco_data.get("annotations", []))
    image_entries = list(coco_data.get("images", []))

    # Prefix each file_name with "images/" so the server's path stripping works.
    for img in image_entries:
        if not img["file_name"].startswith("images/"):
            img["file_name"] = "images/" + img["file_name"]
    coco_data["images"] = image_entries

    dest.parent.mkdir(parents=True, exist_ok=True)

    if progress:
        progress(f"Writing {dest.name}…")

    image_root = training_image_root(project)
    n_written = 0
    n_missing = 0
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("result.json", json.dumps(coco_data))
        for img in image_entries:
            arcname = img["file_name"]
            src = image_root / Path(arcname).name
            if not src.exists():
                n_missing += 1
                if progress:
                    progress(f"  missing image, skipping: {src.name}")
                continue
            zf.write(src, arcname)
            n_written += 1

    return {
        "dest": str(dest),
        "size_bytes": dest.stat().st_size,
        "n_images": n_written,
        "n_missing_images": n_missing,
        "n_annotations": n_anns,
    }
