from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal

from .project import MaterialProject

PROGRESS_PREFIX = "PROGRESS "

RESOURCE_ROOT = Path(__file__).parent.parent / "resources"


def amm_config_path() -> Path:
    return RESOURCE_ROOT / "configs" / "AMM" / "default_config.json"


def gmm_config_path() -> Path:
    return RESOURCE_ROOT / "configs" / "GMM" / "default_config.json"


def m2f_config_path() -> Path:
    return RESOURCE_ROOT / "configs" / "M2F" / "base_config.yaml"


def coco_paths_for(project: MaterialProject) -> dict[str, Path]:
    coco = project.path / "coco"
    return {
        "train": coco / "train_annotations_with_class.json",
        "val": coco / "val_annotations_with_class.json",
    }


def coco_paths_for_segmentation(project: MaterialProject) -> dict[str, Path]:
    coco = project.path / "coco"
    return {
        "train": coco / "train_annotations.json",
        "val": coco / "val_annotations.json",
    }


def coco_files_ready(project: MaterialProject) -> tuple[bool, str]:
    paths = coco_paths_for(project)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        return False, "Run COCO Conversion first (missing: " + ", ".join(
            Path(p).name for p in missing
        ) + ")"
    return True, ""


def coco_files_ready_segmentation(project: MaterialProject) -> tuple[bool, str]:
    paths = coco_paths_for_segmentation(project)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        return False, "Run COCO Conversion first (missing: " + ", ".join(
            Path(p).name for p in missing
        ) + ")"
    return True, ""


def output_dir_for(project: MaterialProject, model: str) -> Path:
    return project.path / "outputs" / model.lower()


def training_image_root(project: MaterialProject) -> Path:
    """Path passed as --train-image-dir / --train-image-root to the runners.

    Points at the staged (flatfield-corrected) copies in coco/images/ when
    COCO Conversion has produced them, otherwise falls back to the raw
    project root.
    """
    coco_images = project.path / "coco" / "images"
    if coco_images.exists() and any(coco_images.iterdir()):
        return coco_images
    return project.path


class TrainingRunner(QObject):
    progress = Signal(dict)
    log = Signal(str)
    finished = Signal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process: QProcess | None = None
        self._buffer = ""

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() != QProcess.NotRunning

    def start_amm(
        self, project: MaterialProject, device: str = "auto"
    ) -> None:
        self._start_module(
            "maskterial_trainer.runners.train_amm",
            project,
            amm_config_path(),
            output_dir_for(project, "amm"),
            extra_args=["--device", device],
        )

    def start_gmm(self, project: MaterialProject) -> None:
        self._start_module(
            "maskterial_trainer.runners.train_gmm",
            project,
            gmm_config_path(),
            output_dir_for(project, "gmm"),
        )

    def start_segmentation(
        self,
        project: MaterialProject,
        pretrained_path: Path,
        max_iter: int = 500,
        ims_per_batch: int = 2,
        base_lr: float = 0.00001,
    ) -> None:
        if self.is_running():
            return
        save_dir = output_dir_for(project, "segmentation")
        save_dir.mkdir(parents=True, exist_ok=True)
        coco = coco_paths_for_segmentation(project)
        args = [
            "-u",
            "-m",
            "maskterial_trainer.runners.train_segmentation",
            "--config-file", str(m2f_config_path()),
            "--train-image-root", str(training_image_root(project)),
            "--train-annotation-path", str(coco["train"]),
            "--output-dir", str(save_dir),
            "--pretrained-weights", str(pretrained_path),
            "--max-iter", str(max_iter),
            "--ims-per-batch", str(ims_per_batch),
            "--base-lr", str(base_lr),
            "--num-gpus", "1",
        ]
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_stdout)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        self._process = proc
        self._buffer = ""
        proc.start(sys.executable, args)

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        if not self._process.waitForFinished(2000):
            self._process.kill()

    def _start_module(
        self,
        module: str,
        project: MaterialProject,
        config: Path,
        save_dir: Path,
        extra_args: list[str] | None = None,
    ) -> None:
        if self.is_running():
            return
        coco = coco_paths_for(project)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Note: val data is intentionally not passed because of an upstream
        # bug in maskterial.utils.data_loader.ContrastDataloader (line ~68)
        # that overwrites X_train when test_* args are set. Evaluation will
        # be a separate step.
        args = [
            "-u",
            "-m",
            module,
            "--config", str(config),
            "--train-image-dir", str(training_image_root(project)),
            "--train-annotation-path", str(coco["train"]),
            "--save-dir", str(save_dir),
            "--seed", "42",
        ]
        if extra_args:
            args.extend(extra_args)
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_stdout)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        self._process = proc
        self._buffer = ""
        proc.start(sys.executable, args)

    def _on_stdout(self) -> None:
        if self._process is None:
            return
        chunk = bytes(self._process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        self._buffer += chunk
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._handle_line(line.rstrip("\r"))

    def _handle_line(self, line: str) -> None:
        if line.startswith(PROGRESS_PREFIX):
            try:
                payload = json.loads(line[len(PROGRESS_PREFIX):])
                self.progress.emit(payload)
                return
            except json.JSONDecodeError:
                pass
        if line:
            self.log.emit(line)

    def _on_finished(self, exit_code: int, _exit_status) -> None:
        if self._buffer:
            self._handle_line(self._buffer)
            self._buffer = ""
        self.finished.emit(exit_code)
        self._process = None

    def _on_error(self, error) -> None:
        self.log.emit(f"[process error] {error}")
