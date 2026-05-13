"""Single-image inference runner using only the AMM/GMM classifier head.

Mirrors the classification-only path inside MaskTerial.predict() but lives
outside detectron2 so it can run without a trained M2F segmentation model.
Writes an annotated PNG overlay (contours + bbox + class label) and a small
JSON sidecar describing each detection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._progress import emit_done, emit_error, emit_progress


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (0, 255, 0)
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["amm", "gmm"], required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--output-image", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--size-threshold", type=int, default=200)
    parser.add_argument(
        "--min-class-occupancy",
        type=float,
        default=0.5,
        help="Ignored in classification-only mode; reserved for future use.",
    )
    parser.add_argument(
        "--class-info",
        default="{}",
        help=(
            'JSON map "{\\"1\\": {\\"name\\": \\"Mono\\", \\"color\\": \\"#ff0000\\"}}". '
            "Used for overlay labels/colors."
        ),
    )
    args = parser.parse_args()

    try:
        import cv2
        import numpy as np
        import torch
    except ImportError as e:
        emit_error(f"Missing dependency: {e}.")
        return 2

    try:
        from ..pipeline.contrasts import compute_contrast_image
    except ImportError as e:
        emit_error(f"Internal import failed: {e}")
        return 2

    emit_progress(0, 4, stage="load_model", message="Loading classifier…")

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

    emit_progress(1, 4, stage="load_image", message="Reading image…")
    image = cv2.imread(args.image_path)
    if image is None:
        emit_error(f"Could not read image: {args.image_path}")
        return 1

    contrast = compute_contrast_image(image)
    if contrast is None:
        emit_error(
            "Could not determine a valid background color for this image "
            "(background appears too dark). Try a different image."
        )
        return 1

    emit_progress(2, 4, stage="classify", message="Running classifier…")
    if args.model_type == "amm":
        contrast_t = torch.from_numpy(contrast).float()
        sem_seg = head(contrast_t).cpu().numpy()
    else:
        sem_seg = head(contrast).cpu().numpy()

    try:
        class_info = json.loads(args.class_info or "{}")
    except json.JSONDecodeError:
        class_info = {}

    default_colors_bgr = [
        (0, 0, 255),    # red
        (255, 0, 0),    # blue
        (0, 255, 0),    # green
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
        (255, 255, 0),  # cyan
    ]

    def color_for(cls: int) -> tuple[int, int, int]:
        info = class_info.get(str(cls))
        if info and "color" in info:
            return _hex_to_bgr(info["color"])
        return default_colors_bgr[cls % len(default_colors_bgr)]

    def name_for(cls: int) -> str:
        info = class_info.get(str(cls))
        if info and "name" in info:
            return info["name"]
        return f"Class {cls}"

    emit_progress(3, 4, stage="draw", message="Drawing overlay…")

    overlay = image.copy()
    detections: list[dict] = []

    for class_id in np.unique(sem_seg):
        if int(class_id) == 0:
            continue
        layer_mask = (sem_seg == class_id).astype(np.uint8)
        layer_mask = cv2.morphologyEx(
            layer_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2
        )
        if cv2.countNonZero(layer_mask) < args.size_threshold:
            continue

        n_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(
            layer_mask, connectivity=4
        )
        color = color_for(int(class_id))
        name = name_for(int(class_id))

        for label_id in range(1, n_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < args.size_threshold:
                continue
            comp_mask = (labeled == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(
                comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            cv2.drawContours(overlay, contours, -1, color, 2)
            x, y, w, h = cv2.boundingRect(comp_mask)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

            label = f"{name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_y = y - 5 if y - th - 6 >= 0 else y + h + th + 5
            bg_y1 = text_y - th - 4 if y - th - 6 >= 0 else y + h
            bg_y2 = text_y + 4 if y - th - 6 >= 0 else y + h + th + 8
            cv2.rectangle(overlay, (x, bg_y1), (x + tw + 6, bg_y2), color, -1)
            cv2.putText(
                overlay,
                label,
                (x + 3, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

            detections.append(
                {
                    "class_id": int(class_id),
                    "class_name": name,
                    "area_px": area,
                    "bbox_xywh": [int(x), int(y), int(w), int(h)],
                }
            )

    out_image = Path(args.output_image)
    out_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_image), overlay)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "image_path": args.image_path,
                "model_type": args.model_type,
                "model_dir": args.model_dir,
                "size_threshold": args.size_threshold,
                "n_detections": len(detections),
                "detections": detections,
            },
            indent=2,
        )
    )

    emit_done(
        f"{len(detections)} detection"
        f"{'s' if len(detections) != 1 else ''} written to {out_image.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
