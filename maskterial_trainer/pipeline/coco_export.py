from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .annotation_io import load_instance_mask, mask_path_for_image
from .flatfield import load_flatfield, remove_vignette
from .project import MaterialProject
from .semantic_io import load_semantic_mask, semantic_mask_path_for_image

DEFAULT_MIN_INSTANCE_SIZE = 300
SPLIT_FILE = "split.json"
VAL_RATIO = 0.2
SPLIT_SEED = 42


def split_dataset(
    image_names: list[str], split_path: Path
) -> dict[str, str]:
    if split_path.exists():
        existing = json.loads(split_path.read_text())
    else:
        existing = {}

    new_names = [n for n in image_names if n not in existing]
    if new_names:
        if not existing:
            rng = random.Random(SPLIT_SEED)
            shuffled = list(new_names)
            rng.shuffle(shuffled)
            n_val = (
                max(1, int(round(len(shuffled) * VAL_RATIO)))
                if len(shuffled) >= 2
                else 0
            )
            for i, name in enumerate(shuffled):
                existing[name] = "val" if i < n_val else "train"
        else:
            for name in new_names:
                h = hashlib.md5(name.encode()).hexdigest()
                existing[name] = "val" if int(h, 16) % 5 == 0 else "train"

    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(existing, indent=2, sort_keys=True))
    return existing


def _encode_rle(mask: np.ndarray) -> dict:
    from pycocotools.mask import encode

    rle = encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def export_coco(
    project: MaterialProject,
    min_instance_size: int = DEFAULT_MIN_INSTANCE_SIZE,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    images = project.list_raw_images()
    if not images:
        return {"error": "No images in project."}

    split_path = project.path / "splits" / SPLIT_FILE
    split = split_dataset([p.name for p in images], split_path)

    records: list[dict] = []
    for i, image_path in enumerate(images):
        if progress is not None:
            progress(i, len(images), image_path.name)
        instance_mask_path = mask_path_for_image(project.path, image_path)
        if not instance_mask_path.exists():
            continue
        instance = load_instance_mask(instance_mask_path)
        if instance is None:
            continue
        sem = load_semantic_mask(
            semantic_mask_path_for_image(project.path, image_path)
        )
        if sem is not None and sem.shape != instance.shape:
            sem = None

        h, w = instance.shape
        instances: list[dict] = []
        for label in np.unique(instance):
            if label == 0:
                continue
            mask = (instance == label).astype(np.uint8)
            size = int(np.count_nonzero(mask))
            if size < min_instance_size:
                continue
            ys, xs = np.where(mask > 0)
            bbox = [
                int(xs.min()),
                int(ys.min()),
                int(xs.max() - xs.min() + 1),
                int(ys.max() - ys.min() + 1),
            ]
            cls_id = 0
            if sem is not None:
                vals, counts = np.unique(sem[mask > 0], return_counts=True)
                cls_id = int(vals[np.argmax(counts)])
            instances.append(
                {
                    "rle": _encode_rle(mask),
                    "bbox": bbox,
                    "area": size,
                    "class_id": cls_id,
                }
            )

        records.append(
            {
                "image_path": image_path,
                "split": split.get(image_path.name, "train"),
                "width": w,
                "height": h,
                "instances": instances,
            }
        )

    if progress is not None:
        progress(len(images), len(images), "")

    if not records:
        return {"error": "No images with instance masks found."}

    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "val"]

    coco_dir = project.path / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)

    # Stage corrected (or copied) images into coco/images/. This is the
    # image_root the training and evaluation runners read from, so the
    # COCO file's plain basename file_names resolve there.
    flatfield_applied = _stage_images(project, records, progress)

    files = {
        "train_annotations.json": _build_coco(train_records, project, with_class=False),
        "train_annotations_with_class.json": _build_coco(
            train_records, project, with_class=True
        ),
        "val_annotations.json": _build_coco(val_records, project, with_class=False),
        "val_annotations_with_class.json": _build_coco(
            val_records, project, with_class=True
        ),
    }
    for name, data in files.items():
        (coco_dir / name).write_text(json.dumps(data))

    return {
        "n_images": len(records),
        "n_train": len(train_records),
        "n_val": len(val_records),
        "n_instances": sum(len(r["instances"]) for r in records),
        "n_class_instances": sum(
            sum(1 for inst in r["instances"] if inst["class_id"] > 0)
            for r in records
        ),
        "coco_dir": str(coco_dir),
        "split_path": str(split_path),
        "flatfield_applied": flatfield_applied,
        "image_root": str(project.path / "coco" / "images"),
    }


def _stage_images(
    project: MaterialProject,
    records: list[dict],
    progress: Callable[[int, int, str], None] | None,
) -> bool:
    """Copy or flatfield-correct each referenced image into coco/images/.

    Returns True if flatfield correction was applied to at least one image.
    """
    images_dir = project.path / "coco" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    existing = {p.name for p in images_dir.iterdir() if p.is_file()}
    wanted = {r["image_path"].name for r in records}
    for stale in existing - wanted:
        try:
            (images_dir / stale).unlink()
        except OSError:
            pass

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

    applied = False
    n = len(records)
    for i, r in enumerate(records):
        src: Path = r["image_path"]
        dest = images_dir / src.name
        if progress is not None:
            progress(i, n, f"Staging {src.name}")
        if flatfield_img is not None:
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is not None and img.shape == flatfield_img.shape:
                corrected = remove_vignette(img, flatfield_img)
                cv2.imwrite(str(dest), corrected)
                applied = True
                continue
        shutil.copy2(src, dest)
    return applied


def _build_coco(
    records: list[dict],
    project: MaterialProject,
    with_class: bool,
) -> dict:
    if with_class:
        categories = [
            {"id": c.id, "name": c.name, "supercategory": c.name}
            for c in project.classes
        ]
    else:
        categories = [{"id": 1, "name": "flake", "supercategory": "flake"}]

    out = {"images": [], "annotations": [], "categories": categories}
    ann_id = 1
    for image_idx, r in enumerate(records, start=1):
        out["images"].append(
            {
                "id": image_idx,
                "file_name": r["image_path"].name,
                "width": r["width"],
                "height": r["height"],
            }
        )
        for inst in r["instances"]:
            if with_class and inst["class_id"] == 0:
                continue
            out["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_idx,
                    "category_id": inst["class_id"] if with_class else 1,
                    "segmentation": inst["rle"],
                    "bbox": inst["bbox"],
                    "area": inst["area"],
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return out
