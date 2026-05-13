"""GMM training runner — spawned as a subprocess by the training tab.

Mirrors MaskTerial_Repo/train_GMM_head.py but emits structured PROGRESS lines
and writes outputs into the project's outputs/gmm/ folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._progress import emit_done, emit_error, emit_progress


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-image-dir", required=True)
    parser.add_argument("--train-annotation-path", required=True)
    parser.add_argument("--val-image-dir")
    parser.add_argument("--val-annotation-path")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        from maskterial.utils.data_loader import ContrastDataloader
    except ImportError as e:
        emit_error(
            f"Missing dependency: {e}. Install MaskTerial "
            "(e.g. `pip install -e .` in the MaskTerial source tree)."
        )
        return 2

    with open(args.config) as f:
        CONFIG = json.load(f)
    DATA_PARAMS = CONFIG["data_params"]
    synthetic_bg_std = float(CONFIG.get("synthetic_background_std", 0.01))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    emit_progress(0, 3, stage="loading", message="Loading dataset…")

    try:
        dataloader = ContrastDataloader(
            train_image_dir=args.train_image_dir,
            train_annotation_path=args.train_annotation_path,
            test_image_dir=args.val_image_dir,
            test_annotation_path=args.val_annotation_path,
            **DATA_PARAMS,
            verbose=True,
        )
    except Exception as e:
        emit_error(f"Failed to load dataset: {e}")
        return 1

    present_class_ids = sorted(int(c) for c in np.unique(dataloader.y_train))
    if not present_class_ids:
        emit_error("No labelled samples were loaded from the dataset.")
        return 1

    emit_progress(1, 3, stage="train", message="Fitting Gaussians…")

    X_train_full = torch.tensor(dataloader.X_train).float()
    y_train_full = dataloader.y_train

    fitted_loc = {
        c: torch.mean(X_train_full[y_train_full == c], dim=0)
        for c in present_class_ids
    }
    fitted_cov = {
        c: torch.cov(X_train_full[y_train_full == c].T)
        for c in present_class_ids
    }

    # Background contrast is (0,0,0) by construction (contrast = image/bg - 1),
    # so synthesize class 0 if it didn't survive denoising or was never sampled.
    # This lets the GMM train from a single annotated layer class.
    if 0 not in fitted_loc:
        emit_progress(
            1, 3, stage="train",
            message="Synthesizing background class at (0,0,0)…",
        )
        fitted_loc[0] = torch.zeros(3, dtype=torch.float32)
        fitted_cov[0] = torch.eye(3, dtype=torch.float32) * (synthetic_bg_std ** 2)

    num_classes = max(fitted_loc.keys()) + 1
    feat_dim = fitted_loc[present_class_ids[0]].shape[0]
    loc_rows = []
    cov_rows = []
    for c in range(num_classes):
        if c in fitted_loc:
            loc_rows.append(fitted_loc[c])
            cov_rows.append(fitted_cov[c])
        else:
            # Class id in the gap (e.g. labels [0,2] with no 1) — fill with NaN
            # so it can't be silently used as a real class downstream.
            loc_rows.append(torch.full((feat_dim,), float("nan")))
            cov_rows.append(torch.full((feat_dim, feat_dim), float("nan")))
    loc = torch.stack(loc_rows)
    cov = torch.stack(cov_rows)

    emit_progress(2, 3, stage="finalizing", message="Saving outputs…")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    synthesized_background = 0 not in present_class_ids
    meta_data = {
        "train_config": CONFIG,
        "train_image_dir": args.train_image_dir,
        "train_annotation_path": args.train_annotation_path,
        "test_image_dir": args.val_image_dir,
        "test_annotation_path": args.val_annotation_path,
        "num_classes": int(num_classes),
        "labelled_class_ids": present_class_ids,
        "synthesized_background": synthesized_background,
        "synthetic_background_std": synthetic_bg_std if synthesized_background else None,
    }
    (save_dir / "meta_data.json").write_text(json.dumps(meta_data, indent=4))

    np.save(save_dir / "loc.npy", loc.numpy())
    np.save(save_dir / "cov.npy", cov.numpy())

    output_dict: dict[str, dict] = {}
    keys = ["b", "g", "r"]
    for c in range(num_classes):
        if c not in fitted_loc:
            continue
        class_loc = loc[c].numpy().tolist()
        class_cov = cov[c].numpy().tolist()
        output_dict[str(c)] = {
            "contrast": {k: class_loc[i] for i, k in enumerate(keys)},
            "covariance_matrix": class_cov,
        }
    (save_dir / "contrast_dict.json").write_text(json.dumps(output_dict, indent=4))

    suffix = " (background synthesized)" if synthesized_background else ""
    emit_done(f"GMM saved to {save_dir} ({num_classes} classes){suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
