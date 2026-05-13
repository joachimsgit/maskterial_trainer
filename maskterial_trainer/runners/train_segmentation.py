"""Segmentation (Mask2Former) training runner — spawned as a subprocess.

Wraps MaskTerial_Trainer (which is a detectron2 DefaultTrainer subclass) and
attaches a HookBase that emits PROGRESS lines from inside the training loop.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from types import SimpleNamespace

from ._progress import emit_done, emit_error, emit_progress


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--train-image-root", required=True)
    parser.add_argument("--train-annotation-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretrained-weights", required=True)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--ims-per-batch", type=int, default=2)
    parser.add_argument("--base-lr", type=float, default=0.00001)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--checkpoint-period", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    try:
        from detectron2.data.datasets import register_coco_instances
        from detectron2.engine import HookBase, launch
        from maskterial.modeling.segmentation_models.M2F.maskformer_model import (  # noqa: F401
            MaskFormer,
        )
        from maskterial.modeling.segmentation_models.M2F.modeling import *  # noqa: F401, F403
        from maskterial.utils.dataset_functions import setup_config
        from maskterial.utils.model_trainer import MaskTerial_Trainer
    except ImportError as e:
        emit_error(
            f"Missing dependency: {e}. Install MaskTerial + detectron2 "
            "(e.g. `pip install -e .` in the MaskTerial source tree)."
        )
        return 2

    config_args = SimpleNamespace(
        config_file=args.config_file,
        train_image_root=args.train_image_root,
        train_annotation_path=args.train_annotation_path,
        opts=[
            "OUTPUT_DIR", args.output_dir,
            "MODEL.WEIGHTS", args.pretrained_weights,
            "SOLVER.MAX_ITER", str(args.max_iter),
            "SOLVER.IMS_PER_BATCH", str(args.ims_per_batch),
            "SOLVER.BASE_LR", str(args.base_lr),
            "SOLVER.CHECKPOINT_PERIOD", str(args.checkpoint_period),
            "DATALOADER.NUM_WORKERS", str(args.num_workers),
        ],
        resume=False,
        eval_only=False,
        pretraining_augmentations=False,
        num_gpus=args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
    )

    class _ProgressEmitter(HookBase):
        def __init__(self, total: int, interval: int = 5) -> None:
            super().__init__()
            self.total = total
            self.interval = max(1, interval)

        def after_step(self):
            it = self.trainer.iter + 1
            if it % self.interval == 0 or it == self.total:
                loss = None
                try:
                    storage = self.trainer.storage
                    if "total_loss" in storage.histories():
                        loss = float(storage.history("total_loss").latest())
                except Exception:
                    loss = None
                emit_progress(
                    it,
                    self.total,
                    loss=loss,
                    stage="train",
                    message=(
                        f"iter {it}/{self.total}"
                        + (f"  loss={loss:.4f}" if loss is not None else "")
                    ),
                )

    def _train_fn(cfg_args: SimpleNamespace) -> None:
        register_coco_instances(
            "Maskterial_Dataset",
            {},
            cfg_args.train_annotation_path,
            cfg_args.train_image_root,
        )
        cfg = setup_config(cfg_args)
        trainer = MaskTerial_Trainer(cfg, pretraining_augmentations=False)
        trainer.register_hooks([_ProgressEmitter(cfg.SOLVER.MAX_ITER)])
        trainer.resume_or_load(resume=False)
        trainer.train()

    emit_progress(0, args.max_iter, stage="loading", message="Setting up trainer…")

    try:
        launch(
            _train_fn,
            num_gpus_per_machine=args.num_gpus,
            num_machines=1,
            machine_rank=0,
            dist_url="auto",
            args=(config_args,),
        )
    except Exception as e:
        traceback.print_exc()
        emit_error(f"Training failed: {e}")
        return 1

    emit_done(f"Segmentation model saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
