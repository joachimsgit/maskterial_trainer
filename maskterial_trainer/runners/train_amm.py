"""AMM training runner — spawned as a subprocess by the training tab.

Mirrors MaskTerial_Repo/train_AMM_head.py but reads our bundled config,
writes outputs into the project's outputs/amm/ folder, and emits
structured PROGRESS lines on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device. 'auto' picks CUDA if available, else CPU.",
    )
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from maskterial.modeling.common.fcresnet import FCResNet
        from maskterial.utils.data_loader import ContrastDataloader
    except ImportError as e:
        emit_error(
            f"Missing dependency: {e}. Install MaskTerial "
            "(e.g. `pip install -e .` in the MaskTerial source tree)."
        )
        return 2

    if args.device == "cuda":
        if not torch.cuda.is_available():
            emit_error("CUDA was requested but no CUDA device is available.")
            return 1
        DEVICE = "cuda"
    elif args.device == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = "cpu"
    print(f"Using device: {DEVICE}", flush=True)

    with open(args.config) as f:
        CONFIG = json.load(f)
    TRAIN_PARAMS = CONFIG["train_params"]
    DATA_PARAMS = CONFIG["data_params"]
    MODEL_ARCHITECTURE = CONFIG["model_arch"]
    NUM_ITER = int(TRAIN_PARAMS["num_iterations"])
    TEST_INTERVAL = int(TRAIN_PARAMS["test_interval"])
    LR = float(TRAIN_PARAMS["learning_rate"])
    BS = int(TRAIN_PARAMS["batch_size"])

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    emit_progress(0, NUM_ITER, stage="loading", message="Loading dataset…")

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

    if dataloader.num_classes < 2:
        emit_error(
            f"Need at least 2 classes (background + at least one layer); "
            f"found {dataloader.num_classes}. The ContrastDataloader auto-samples "
            "background pixels, so this usually means every image had an invalid "
            "background colour (too dark or wrong flatfield) and was skipped, "
            "or denoising removed the background class. Inspect the log above "
            "for per-image skip messages."
        )
        return 1

    MODEL_ARCHITECTURE["num_classes"] = dataloader.num_classes

    has_test = bool(args.val_annotation_path and args.val_image_dir)
    if has_test:
        X_test, y_test = dataloader.get_test_data()
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        y_test_t = torch.tensor(y_test, dtype=torch.int64).to(DEVICE)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = FCResNet(**MODEL_ARCHITECTURE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_function = nn.CrossEntropyLoss()

    emit_progress(0, NUM_ITER, stage="train", message="Training…")
    best_loss = float("inf")
    test_loss = float("inf")
    train_loss = float("inf")

    for iteration in range(NUM_ITER):
        model.train()
        optimizer.zero_grad()
        X_batch, y_batch = dataloader.get_batch(batch_size=BS)
        X_batch = torch.tensor(X_batch, dtype=torch.float32).to(DEVICE)
        y_batch = torch.tensor(y_batch, dtype=torch.int64).to(DEVICE)
        logits = model(X_batch)
        loss = loss_function(logits, y_batch)
        loss.backward()
        optimizer.step()

        if iteration % TEST_INTERVAL == 0:
            train_loss = float(loss.item())
            if has_test:
                model.eval()
                with torch.no_grad():
                    test_logits = model(X_test_t)
                    test_loss = float(loss_function(test_logits, y_test_t).item())
                if test_loss < best_loss:
                    best_loss = test_loss
            emit_progress(
                iteration,
                NUM_ITER,
                loss=train_loss,
                stage="train",
                message=(
                    f"train={train_loss:.4f}  test={test_loss:.4f}  best={best_loss:.4f}"
                ),
            )

    if has_test:
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t)
            test_loss = float(loss_function(test_logits, y_test_t).item())
        if test_loss < best_loss:
            best_loss = test_loss

    emit_progress(NUM_ITER, NUM_ITER, stage="finalizing", message="Computing class embeddings…")

    X_train_full = torch.tensor(dataloader.X_train).float()
    y_train_full = dataloader.y_train
    model.cpu()
    model.eval()
    with torch.no_grad():
        X_embeddings = model.get_embedding(X_train_full)

    loc = torch.stack(
        [
            torch.mean(X_embeddings[y_train_full == c], dim=0)
            for c in range(dataloader.num_classes)
        ]
    )
    cov = torch.stack(
        [
            torch.cov(X_embeddings[y_train_full == c].T)
            for c in range(dataloader.num_classes)
        ]
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # X_train_mean / X_train_std are only set by ContrastDataloader when
    # use_normalization=True. Guard so the runner doesn't crash post-training
    # if a user disables it in the config.
    train_mean = getattr(dataloader, "X_train_mean", None)
    train_std = getattr(dataloader, "X_train_std", None)
    meta_data = {
        "train_config": CONFIG,
        "train_mean": train_mean.tolist() if train_mean is not None else None,
        "train_std": train_std.tolist() if train_std is not None else None,
        "train_image_dir": args.train_image_dir,
        "train_annotation_path": args.train_annotation_path,
        "test_image_dir": args.val_image_dir,
        "test_annotation_path": args.val_annotation_path,
        "num_classes": int(dataloader.num_classes),
        "device": DEVICE,
        "final_train_loss": float(train_loss) if train_loss != float("inf") else None,
    }
    if has_test:
        meta_data["test_losses"] = {"final": test_loss, "best": best_loss}
    (save_dir / "meta_data.json").write_text(
        json.dumps(meta_data, indent=4, allow_nan=False)
    )

    np.save(save_dir / "loc.npy", loc.numpy())
    np.save(save_dir / "cov.npy", cov.numpy())
    torch.save(model.state_dict(), save_dir / "model.pth")

    emit_done(
        f"AMM saved to {save_dir} (best test loss: {best_loss:.4f})"
        if has_test
        else f"AMM saved to {save_dir} (final train loss: {train_loss:.4f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
