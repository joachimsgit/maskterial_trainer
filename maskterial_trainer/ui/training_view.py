from __future__ import annotations

import time

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..pipeline.model_download import (
    PRETRAINED_M2F_PATH,
    download_pretrained_m2f,
    pretrained_m2f_available,
)
from ..pipeline.project import MaterialProject
from ..pipeline.training import (
    TrainingRunner,
    coco_files_ready,
    coco_files_ready_segmentation,
    output_dir_for,
)
from ..pipeline.training_zip import build_training_zip


def _format_eta(seconds: float) -> str:
    if seconds < 0 or seconds == float("inf"):
        return "—"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def _format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    f = float(n)
    i = 0
    while f >= 1024 and i < len(units) - 1:
        f /= 1024
        i += 1
    return f"{f:.1f} {units[i]}"


class _DownloadWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(bool, str)

    def run(self) -> None:
        try:
            path = download_pretrained_m2f(progress=self._on_progress)
            self.finished.emit(True, str(path))
        except Exception as e:
            self.finished.emit(False, f"{type(e).__name__}: {e}")

    def _on_progress(self, downloaded: int, total: int, stage: str) -> None:
        self.progress.emit(downloaded, total, stage)


class TrainingView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.project: MaterialProject | None = None
        self._start_time: float | None = None
        self._first_progress_step: int | None = None

        self.runner = TrainingRunner(self)
        self.runner.progress.connect(self._on_progress)
        self.runner.log.connect(self._on_log)
        self.runner.finished.connect(self._on_finished)

        self._download_thread: QThread | None = None
        self._download_worker: _DownloadWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Train model")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        explainer = QLabel(
            "Pick a model and start training. AMM/GMM read "
            "coco/train_annotations_with_class.json. Segmentation reads "
            "coco/train_annotations.json (no class info — segmentation only "
            "differentiates flake from background; class assignment is "
            "what AMM/GMM are for). Outputs go to outputs/<model>/."
        )
        explainer.setWordWrap(True)
        explainer.setStyleSheet("color: #555;")
        layout.addWidget(explainer)
        layout.addSpacing(8)

        # Model picker
        picker_row = QHBoxLayout()
        picker_row.addWidget(QLabel("Model:"))
        self.amm_radio = QRadioButton("AMM (deep classifier)")
        self.gmm_radio = QRadioButton("GMM (Gaussian per class)")
        self.seg_radio = QRadioButton("Segmentation (Mask2Former, GPU required)")
        self.amm_radio.setChecked(True)
        self._model_group = QButtonGroup(self)
        self._model_group.addButton(self.amm_radio)
        self._model_group.addButton(self.gmm_radio)
        self._model_group.addButton(self.seg_radio)
        self.amm_radio.toggled.connect(self._on_model_changed)
        self.gmm_radio.toggled.connect(self._on_model_changed)
        self.seg_radio.toggled.connect(self._on_model_changed)
        picker_row.addWidget(self.amm_radio)
        picker_row.addWidget(self.gmm_radio)
        picker_row.addWidget(self.seg_radio)
        picker_row.addStretch(1)
        layout.addLayout(picker_row)

        # AMM-only: device picker (auto / cpu / cuda). GMM is a closed-form
        # mean/covariance fit so it's already cheap on CPU; segmentation
        # always needs the GPU and is handled separately.
        self.amm_device_row = QWidget()
        device_layout = QHBoxLayout(self.amm_device_row)
        device_layout.setContentsMargins(0, 0, 0, 0)
        device_layout.addWidget(QLabel("AMM device:"))
        self.amm_device_combo = QComboBox()
        self.amm_device_combo.addItem("Auto (CUDA if available)", "auto")
        self.amm_device_combo.addItem("CPU", "cpu")
        self.amm_device_combo.addItem("CUDA (GPU)", "cuda")
        device_layout.addWidget(self.amm_device_combo)
        device_layout.addStretch(1)
        layout.addWidget(self.amm_device_row)

        # Segmentation-specific panel: local-training and server-training subsections
        self.seg_panel = QWidget()
        sp_layout = QVBoxLayout(self.seg_panel)
        sp_layout.setContentsMargins(0, 4, 0, 4)

        local_header = QLabel("Local training (uses this machine's GPU)")
        local_header.setStyleSheet("font-weight: bold; color: #333;")
        sp_layout.addWidget(local_header)
        self.pretrained_label = QLabel("")
        self.pretrained_label.setWordWrap(True)
        sp_layout.addWidget(self.pretrained_label)
        dl_row = QHBoxLayout()
        self.download_btn = QPushButton("Download pretrained backbone (~600 MB)")
        self.download_btn.clicked.connect(self._start_download)
        dl_row.addWidget(self.download_btn)
        dl_row.addStretch(1)
        sp_layout.addLayout(dl_row)
        self.download_progress = QProgressBar()
        self.download_progress.setRange(0, 100)
        self.download_progress.setVisible(False)
        sp_layout.addWidget(self.download_progress)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet("color: #ddd;")
        sp_layout.addWidget(sep)

        server_header = QLabel("Server training (upload to inference website)")
        server_header.setStyleSheet("font-weight: bold; color: #333;")
        sp_layout.addWidget(server_header)
        server_explainer = QLabel(
            "Builds a training zip in the format the website's training "
            "endpoint expects. Save it, then upload via the website to train "
            "on the server's GPU."
        )
        server_explainer.setWordWrap(True)
        server_explainer.setStyleSheet("color: #555;")
        sp_layout.addWidget(server_explainer)
        zip_row = QHBoxLayout()
        self.zip_btn = QPushButton("Build training zip…")
        self.zip_btn.clicked.connect(self._build_training_zip)
        zip_row.addWidget(self.zip_btn)
        zip_row.addStretch(1)
        sp_layout.addLayout(zip_row)

        self.seg_panel.setVisible(False)
        layout.addWidget(self.seg_panel)

        # Status / readiness
        self.readiness_label = QLabel("")
        self.readiness_label.setStyleSheet("color: #b8860b; padding: 4px 0;")
        self.readiness_label.setWordWrap(True)
        layout.addWidget(self.readiness_label)

        # Buttons
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start training")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        # Progress bar + ETA
        prog_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        prog_row.addWidget(self.progress_bar, 1)
        self.eta_label = QLabel("")
        self.eta_label.setMinimumWidth(180)
        self.eta_label.setStyleSheet("color: #555;")
        prog_row.addWidget(self.eta_label)
        layout.addLayout(prog_row)

        self.stage_label = QLabel("")
        self.stage_label.setStyleSheet("color: #555;")
        layout.addWidget(self.stage_label)

        # Log
        layout.addWidget(QLabel("Log"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet(
            "QPlainTextEdit { background: #1e1e1e; color: #d0d0d0; "
            "font-family: Consolas, monospace; font-size: 11px; }"
        )
        layout.addWidget(self.log_view, 1)

    # ---------- public ----------

    def set_project(self, project: MaterialProject) -> None:
        self.project = project
        self._refresh_readiness()
        self._reset_view()

    def refresh(self) -> None:
        self._refresh_readiness()

    # ---------- internals ----------

    def _selected_model(self) -> str:
        if self.gmm_radio.isChecked():
            return "gmm"
        if self.seg_radio.isChecked():
            return "segmentation"
        return "amm"

    def _on_model_changed(self) -> None:
        self.seg_panel.setVisible(self.seg_radio.isChecked())
        self.amm_device_row.setVisible(self.amm_radio.isChecked())
        self._refresh_readiness()

    def _refresh_readiness(self) -> None:
        if self.project is None:
            self.readiness_label.setText("No project loaded.")
            self.start_btn.setEnabled(False)
            return

        model = self._selected_model()
        if model == "segmentation":
            self._update_pretrained_label()
            ok, msg = coco_files_ready_segmentation(self.project)
            if not ok:
                self._set_readiness(msg, ready=False)
                self.start_btn.setEnabled(False)
                return
            if not pretrained_m2f_available():
                self._set_readiness(
                    "Pretrained backbone not downloaded yet.", ready=False
                )
                self.start_btn.setEnabled(False)
                return
            self._set_readiness(
                "COCO files + pretrained backbone found. Ready to train (GPU required).",
                ready=True,
            )
        else:
            ok, msg = coco_files_ready(self.project)
            if not ok:
                self._set_readiness(msg, ready=False)
                self.start_btn.setEnabled(False)
                return
            self._set_readiness("COCO files found. Ready to train.", ready=True)

        self.start_btn.setEnabled(not self.runner.is_running())

    def _set_readiness(self, msg: str, ready: bool) -> None:
        self.readiness_label.setText(msg)
        if ready:
            self.readiness_label.setStyleSheet(
                "color: #2e7d32; padding: 4px 0;"
            )
        else:
            self.readiness_label.setStyleSheet(
                "color: #b8860b; padding: 4px 0;"
            )

    def _update_pretrained_label(self) -> None:
        if pretrained_m2f_available():
            size = PRETRAINED_M2F_PATH.stat().st_size
            self.pretrained_label.setText(
                f"<span style='color:#2e7d32;'>✓ Pretrained backbone ready</span> "
                f"({_format_bytes(size)} at {PRETRAINED_M2F_PATH})"
            )
            self.pretrained_label.setTextFormat(Qt.RichText)
            self.download_btn.setEnabled(False)
            self.download_btn.setText("Re-download pretrained backbone")
            self.download_btn.setEnabled(True)
        else:
            self.pretrained_label.setText(
                "<span style='color:#b8860b;'>✗ Pretrained backbone not downloaded.</span> "
                "First-time segmentation training needs a one-off ~600 MB download "
                "from Zenodo (cached under ~/.maskterial_trainer/models/)."
            )
            self.pretrained_label.setTextFormat(Qt.RichText)
            self.download_btn.setEnabled(True)
            self.download_btn.setText("Download pretrained backbone (~600 MB)")

    def _reset_view(self) -> None:
        self.progress_bar.setValue(0)
        self.eta_label.setText("")
        self.stage_label.setText("")
        self.log_view.clear()

    def _start(self) -> None:
        if self.project is None or self.runner.is_running():
            return
        self._reset_view()
        model = self._selected_model()
        out_dir = output_dir_for(
            self.project, "segmentation" if model == "segmentation" else model
        )
        self._append_log(
            f"=== Training {model.upper()} for project '{self.project.name}' ==="
        )
        self._append_log(f"Output: {out_dir}")
        self._start_time = time.time()
        self._first_progress_step = None
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.amm_radio.setEnabled(False)
        self.gmm_radio.setEnabled(False)
        self.seg_radio.setEnabled(False)
        if model == "amm":
            device = self.amm_device_combo.currentData() or "auto"
            self._append_log(f"AMM device: {device}")
            self.runner.start_amm(self.project, device=device)
        elif model == "gmm":
            self.runner.start_gmm(self.project)
        else:
            self.runner.start_segmentation(
                self.project, PRETRAINED_M2F_PATH
            )

    def _stop(self) -> None:
        if not self.runner.is_running():
            return
        self._append_log("[stopping…]")
        self.runner.stop()

    def _on_progress(self, payload: dict) -> None:
        step = int(payload.get("step", 0))
        total = int(payload.get("total", 0))
        stage = str(payload.get("stage", ""))
        message = str(payload.get("message", ""))

        pct = int(round(100 * step / total)) if total > 0 else 0
        self.progress_bar.setValue(max(0, min(100, pct)))

        if stage == "train":
            if self._first_progress_step is None:
                self._first_progress_step = step
                self._start_time = time.time()
            elif self._start_time is not None and step > self._first_progress_step:
                elapsed = time.time() - self._start_time
                done = step - self._first_progress_step
                remaining = total - step
                rate = done / elapsed if elapsed > 0 else 0
                eta = remaining / rate if rate > 0 else float("inf")
                self.eta_label.setText(
                    f"step {step}/{total} · ETA {_format_eta(eta)}"
                )
            else:
                self.eta_label.setText(f"step {step}/{total}")
        elif stage == "done":
            self.progress_bar.setValue(100)
            self.eta_label.setText("done")
        elif stage == "error":
            self.eta_label.setText("error")

        if message:
            self.stage_label.setText(f"{stage}: {message}")
        else:
            self.stage_label.setText(stage)

    def _on_log(self, line: str) -> None:
        self._append_log(line)

    def _on_finished(self, exit_code: int) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.amm_radio.setEnabled(True)
        self.gmm_radio.setEnabled(True)
        self.seg_radio.setEnabled(True)
        if exit_code == 0:
            self._append_log("=== Training finished successfully ===")
        else:
            self._append_log(f"=== Training exited with code {exit_code} ===")
        self._refresh_readiness()

    def _append_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)
        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    # ---------- pretrained download ----------

    def _start_download(self) -> None:
        if self._download_thread is not None:
            return
        self.download_btn.setEnabled(False)
        self.download_progress.setVisible(True)
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        self._append_log("Downloading pretrained M2F backbone…")

        self._download_thread = QThread(self)
        self._download_worker = _DownloadWorker()
        self._download_worker.moveToThread(self._download_thread)
        self._download_thread.started.connect(self._download_worker.run)
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_thread.start()

    def _on_download_progress(self, downloaded: int, total: int, stage: str) -> None:
        if stage == "extracting":
            self.download_progress.setRange(0, 0)
            self.stage_label.setText("download: extracting…")
            return
        if stage == "done":
            self.download_progress.setRange(0, 100)
            self.download_progress.setValue(100)
            return
        if total > 0:
            self.download_progress.setRange(0, 100)
            pct = int(100 * downloaded / total)
            self.download_progress.setValue(pct)
            self.stage_label.setText(
                f"download: {_format_bytes(downloaded)} / {_format_bytes(total)} "
                f"({pct}%)"
            )

    def _on_download_finished(self, ok: bool, message: str) -> None:
        if self._download_thread is not None:
            self._download_thread.quit()
            self._download_thread.wait()
            self._download_thread = None
        self._download_worker = None
        self.download_progress.setVisible(False)
        self.download_btn.setEnabled(True)
        if ok:
            self._append_log(f"Pretrained backbone downloaded: {message}")
        else:
            self._append_log(f"Download failed: {message}")
        self._refresh_readiness()

    # ---------- training zip (server upload) ----------

    def _build_training_zip(self) -> None:
        if self.project is None:
            return
        coco_path = self.project.path / "coco" / "train_annotations.json"
        if not coco_path.exists():
            self._append_log(
                "Cannot build zip: train_annotations.json missing — "
                "run COCO Conversion first."
            )
            return
        default_path = self.project.path.parent / f"{self.project.name}_train.zip"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save training zip",
            str(default_path),
            "Zip archives (*.zip)",
        )
        if not path:
            return
        self.zip_btn.setEnabled(False)
        self._append_log(f"Building training zip → {path}")
        self.repaint()
        try:
            result = build_training_zip(
                self.project, Path(path), progress=self._append_log
            )
        except Exception as e:
            self._append_log(f"Failed to build zip: {e}")
            self.zip_btn.setEnabled(True)
            return
        size_mb = result["size_bytes"] / (1024 * 1024)
        self._append_log(
            f"Zip ready: {result['dest']} "
            f"({result['n_images']} images · {result['n_annotations']} annotations · "
            f"{size_mb:.1f} MB)"
        )
        if result["n_missing_images"]:
            self._append_log(
                f"  {result['n_missing_images']} image(s) listed in COCO but not "
                "found on disk — they were skipped."
            )
        self._append_log(
            "Upload this zip via the website's segmentation training page."
        )
        self.zip_btn.setEnabled(True)
