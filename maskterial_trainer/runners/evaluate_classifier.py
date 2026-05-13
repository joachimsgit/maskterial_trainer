"""Classifier evaluation runner — spawned as a subprocess by the Evaluate tab.

Loads a trained AMM or GMM head, runs inference on the project's val COCO
file, and writes per-class precision/recall/F1 + a confusion matrix to
outputs/<model>/evaluation.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._progress import emit_done, emit_error, emit_progress


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["amm", "gmm"], required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--annotation-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        import cv2
        import numpy as np
        import torch
        from pycocotools.coco import COCO
    except ImportError as e:
        emit_error(f"Missing dependency: {e}.")
        return 2

    try:
        from ..pipeline.contrasts import calculate_background_color
    except ImportError as e:
        emit_error(f"Internal import failed: {e}")
        return 2

    if args.model_type == "amm":
        try:
            from maskterial.modeling.classification_models.AMM.AMM_head import (
                AMM_head,
            )
        except ImportError as e:
            emit_error(f"Failed to import AMM: {e}. Install MaskTerial.")
            return 2
        try:
            head = AMM_head.from_pretrained(args.model_dir)
        except Exception as e:
            emit_error(f"Failed to load AMM model from {args.model_dir}: {e}")
            return 1
    else:
        try:
            from maskterial.modeling.classification_models.GMM.GMM_head import (
                GMM_head,
            )
        except ImportError as e:
            emit_error(f"Failed to import GMM: {e}. Install MaskTerial.")
            return 2
        try:
            head = GMM_head.from_pretrained(args.model_dir)
        except Exception as e:
            emit_error(f"Failed to load GMM model from {args.model_dir}: {e}")
            return 1

    coco = COCO(args.annotation_path)
    image_ids = coco.getImgIds()
    if not image_ids:
        emit_error("No images in annotation file.")
        return 1

    image_dir = Path(args.image_dir)
    pairs: list[tuple[int, int]] = []
    n_skipped = 0

    for i, img_id in enumerate(image_ids):
        info = coco.loadImgs([img_id])[0]
        emit_progress(
            i, len(image_ids), stage="evaluate", message=info["file_name"]
        )
        img_path = image_dir / info["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            n_skipped += 1
            continue

        bg = calculate_background_color(image)
        if np.any(bg < 1):
            n_skipped += 1
            continue

        contrast = image.astype(np.float32) / bg.astype(np.float32) - 1.0

        if args.model_type == "amm":
            contrast_t = torch.from_numpy(contrast).float()
            pred = head(contrast_t).cpu().numpy()
        else:
            pred = head(contrast).cpu().numpy()

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        for ann in coco.loadAnns(ann_ids):
            gt_class = int(ann["category_id"])
            mask = coco.annToMask(ann)
            if mask.sum() == 0:
                continue
            instance_pred_pixels = pred[mask > 0]
            if len(instance_pred_pixels) == 0:
                continue
            vals, counts = np.unique(instance_pred_pixels, return_counts=True)
            predicted_class = int(vals[np.argmax(counts)])
            pairs.append((gt_class, predicted_class))

    emit_progress(
        len(image_ids), len(image_ids), stage="finalizing", message="Computing metrics…"
    )

    if not pairs:
        emit_error(
            "No instances were evaluated. Make sure your val split has "
            "annotated flakes (try re-running COCO Conversion)."
        )
        return 1

    gt = np.array([p[0] for p in pairs])
    pr = np.array([p[1] for p in pairs])

    classes = sorted(set(int(c) for c in gt.tolist()) | set(int(c) for c in pr.tolist()) | {0})
    idx_of = {c: i for i, c in enumerate(classes)}

    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for g, p in zip(gt, pr):
        cm[idx_of[int(g)], idx_of[int(p)]] += 1

    per_class: dict[str, dict] = {}
    for cls in classes:
        tp = int(((gt == cls) & (pr == cls)).sum())
        fp = int(((gt != cls) & (pr == cls)).sum())
        fn = int(((gt == cls) & (pr != cls)).sum())
        support = int((gt == cls).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[str(cls)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }

    accuracy = float((gt == pr).sum() / len(gt))

    result = {
        "model_type": args.model_type,
        "model_dir": args.model_dir,
        "annotation_path": args.annotation_path,
        "n_instances": len(pairs),
        "n_skipped_images": n_skipped,
        "accuracy": accuracy,
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False))

    emit_done(
        f"Accuracy {accuracy:.3f} on {len(pairs)} instance"
        f"{'s' if len(pairs) != 1 else ''}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
