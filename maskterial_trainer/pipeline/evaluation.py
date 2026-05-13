from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal

from .project import MaterialProject
from .training import training_image_root

PROGRESS_PREFIX = "PROGRESS "


def amm_model_dir(project: MaterialProject) -> Path:
    return project.path / "outputs" / "amm"


def gmm_model_dir(project: MaterialProject) -> Path:
    return project.path / "outputs" / "gmm"


def amm_available(project: MaterialProject) -> bool:
    return (amm_model_dir(project) / "model.pth").exists()


def gmm_available(project: MaterialProject) -> bool:
    return (gmm_model_dir(project) / "contrast_dict.json").exists()


def evaluation_path(project: MaterialProject, model: str) -> Path:
    return project.path / "outputs" / model / "evaluation.json"


def val_annotation_path(project: MaterialProject) -> Path:
    return project.path / "coco" / "val_annotations_with_class.json"


def val_available(project: MaterialProject) -> bool:
    return val_annotation_path(project).exists()


def inference_output_dir(project: MaterialProject) -> Path:
    return project.path / "outputs" / "inference_previews"


class EvaluationRunner(QObject):
    progress = Signal(dict)
    log = Signal(str)
    finished = Signal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process: QProcess | None = None
        self._buffer = ""

    def is_running(self) -> bool:
        return (
            self._process is not None
            and self._process.state() != QProcess.NotRunning
        )

    def start(self, project: MaterialProject, model: str) -> None:
        if self.is_running() or model not in {"amm", "gmm"}:
            return
        mdir = amm_model_dir(project) if model == "amm" else gmm_model_dir(project)
        out_path = evaluation_path(project, model)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ann = val_annotation_path(project)
        args = [
            "-u",
            "-m",
            "maskterial_trainer.runners.evaluate_classifier",
            "--model-type", model,
            "--model-dir", str(mdir),
            "--image-dir", str(training_image_root(project)),
            "--annotation-path", str(ann),
            "--output", str(out_path),
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


class InferenceRunner(QObject):
    """Runs a single-image AMM/GMM inference and writes an overlay PNG."""

    progress = Signal(dict)
    log = Signal(str)
    finished = Signal(int, str, str)  # exit_code, image_path, json_path

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process: QProcess | None = None
        self._buffer = ""
        self._out_image = ""
        self._out_json = ""

    def is_running(self) -> bool:
        return (
            self._process is not None
            and self._process.state() != QProcess.NotRunning
        )

    def start(
        self,
        project: MaterialProject,
        model: str,
        image_path: Path,
        size_threshold: int = 200,
        class_info_json: str = "{}",
    ) -> None:
        if self.is_running() or model not in {"amm", "gmm"}:
            return
        mdir = amm_model_dir(project) if model == "amm" else gmm_model_dir(project)
        preview_dir = inference_output_dir(project)
        preview_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        out_image = preview_dir / f"{stem}__{model}.png"
        out_json = preview_dir / f"{stem}__{model}.json"
        self._out_image = str(out_image)
        self._out_json = str(out_json)

        args = [
            "-u",
            "-m",
            "maskterial_trainer.runners.run_inference",
            "--model-type", model,
            "--model-dir", str(mdir),
            "--image-path", str(image_path),
            "--output-image", str(out_image),
            "--output-json", str(out_json),
            "--size-threshold", str(size_threshold),
            "--class-info", class_info_json,
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
        self.finished.emit(exit_code, self._out_image, self._out_json)
        self._process = None

    def _on_error(self, error) -> None:
        self.log.emit(f"[process error] {error}")
