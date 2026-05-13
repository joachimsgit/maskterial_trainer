from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..pipeline.evaluation import (
    EvaluationRunner,
    InferenceRunner,
    amm_available,
    evaluation_path,
    gmm_available,
    inference_output_dir,
    val_annotation_path,
    val_available,
)
from ..pipeline.project import MaterialProject
from ..pipeline.training import training_image_root

LOW_SUPPORT_THRESHOLD = 10


class EvaluationView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.project: MaterialProject | None = None

        self.runner = EvaluationRunner(self)
        self.runner.progress.connect(self._on_progress)
        self.runner.log.connect(self._on_log)
        self.runner.finished.connect(self._on_finished)

        self.inference_runner = InferenceRunner(self)
        self.inference_runner.progress.connect(self._on_inference_progress)
        self.inference_runner.log.connect(self._on_log)
        self.inference_runner.finished.connect(self._on_inference_finished)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Evaluate classifier")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        outer.addWidget(title)

        explainer = QLabel(
            "Runs the trained classifier on the val split "
            "(coco/val_annotations_with_class.json) and reports per-class "
            "precision / recall / F1 plus a confusion matrix. The result is "
            "saved to outputs/<model>/evaluation.json so it travels with the "
            "model bundle."
        )
        explainer.setWordWrap(True)
        explainer.setStyleSheet("color: #555;")
        outer.addWidget(explainer)
        outer.addSpacing(8)

        # Model picker (shared by metrics + inference)
        picker_row = QHBoxLayout()
        picker_row.addWidget(QLabel("Model:"))
        self.amm_radio = QRadioButton("AMM")
        self.gmm_radio = QRadioButton("GMM")
        self._model_group = QButtonGroup(self)
        self._model_group.addButton(self.amm_radio)
        self._model_group.addButton(self.gmm_radio)
        picker_row.addWidget(self.amm_radio)
        picker_row.addWidget(self.gmm_radio)
        picker_row.addStretch(1)
        outer.addLayout(picker_row)

        self.readiness_label = QLabel("")
        self.readiness_label.setWordWrap(True)
        outer.addWidget(self.readiness_label)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_metrics_panel())
        splitter.addWidget(self._build_inference_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, 1)

        # Shared log
        outer.addWidget(QLabel("Log"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(140)
        self.log_view.setStyleSheet(
            "QPlainTextEdit { background: #1e1e1e; color: #d0d0d0; "
            "font-family: Consolas, monospace; font-size: 11px; }"
        )
        outer.addWidget(self.log_view)

    # ---------- panel builders ----------

    def _build_metrics_panel(self) -> QWidget:
        box = QGroupBox("Metrics (val split)")
        layout = QVBoxLayout(box)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run evaluation")
        self.run_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        prog_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        prog_row.addWidget(self.progress_bar, 1)
        self.stage_label = QLabel("")
        self.stage_label.setStyleSheet("color: #555;")
        self.stage_label.setMinimumWidth(220)
        prog_row.addWidget(self.stage_label)
        layout.addLayout(prog_row)

        self.headline_label = QLabel("")
        self.headline_label.setStyleSheet(
            "font-size: 22px; font-weight: bold; padding: 6px 0;"
        )
        layout.addWidget(self.headline_label)

        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: #b8860b;")
        self.warning_label.setWordWrap(True)
        layout.addWidget(self.warning_label)

        layout.addWidget(QLabel("Per-class metrics"))
        self.per_class_table = QTableWidget(0, 5)
        self.per_class_table.setHorizontalHeaderLabels(
            ["Class", "Support", "Precision", "Recall", "F1"]
        )
        self.per_class_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.per_class_table.verticalHeader().setVisible(False)
        layout.addWidget(self.per_class_table)

        layout.addWidget(
            QLabel("Confusion matrix (rows = ground truth, cols = predicted)")
        )
        self.cm_table = QTableWidget(0, 0)
        self.cm_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        layout.addWidget(self.cm_table)

        return box

    def _build_inference_panel(self) -> QWidget:
        box = QGroupBox("Inference preview (classifier-only)")
        layout = QVBoxLayout(box)

        hint = QLabel(
            "Runs the selected classifier on a single image using AMM/GMM "
            "as a per-pixel labeller (no M2F segmentation required). "
            "Connected components above the size threshold become detections."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555;")
        layout.addWidget(hint)

        pick_row = QHBoxLayout()
        pick_row.addWidget(QLabel("Image:"))
        self.image_combo = QComboBox()
        self.image_combo.setMinimumWidth(220)
        pick_row.addWidget(self.image_combo, 1)
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._browse_image)
        pick_row.addWidget(self.browse_btn)
        layout.addLayout(pick_row)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Min size (px):"))
        self.size_threshold = QSpinBox()
        self.size_threshold.setRange(1, 100000)
        self.size_threshold.setValue(200)
        self.size_threshold.setSingleStep(50)
        size_row.addWidget(self.size_threshold)
        size_row.addStretch(1)
        self.infer_btn = QPushButton("Run inference")
        self.infer_btn.clicked.connect(self._start_inference)
        size_row.addWidget(self.infer_btn)
        self.infer_stop_btn = QPushButton("Stop")
        self.infer_stop_btn.clicked.connect(self._stop_inference)
        self.infer_stop_btn.setEnabled(False)
        size_row.addWidget(self.infer_stop_btn)
        layout.addLayout(size_row)

        prog_row = QHBoxLayout()
        self.infer_progress = QProgressBar()
        self.infer_progress.setRange(0, 100)
        prog_row.addWidget(self.infer_progress, 1)
        self.infer_stage_label = QLabel("")
        self.infer_stage_label.setStyleSheet("color: #555;")
        self.infer_stage_label.setMinimumWidth(180)
        prog_row.addWidget(self.infer_stage_label)
        layout.addLayout(prog_row)

        self.infer_summary_label = QLabel("")
        self.infer_summary_label.setStyleSheet(
            "font-weight: bold; padding: 4px 0;"
        )
        layout.addWidget(self.infer_summary_label)

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setAlignment(Qt.AlignCenter)
        self.preview_scroll.setStyleSheet(
            "QScrollArea { background: #222; }"
        )
        self.preview_label = QLabel("No preview yet.")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("color: #aaa; padding: 24px;")
        self.preview_scroll.setWidget(self.preview_label)
        layout.addWidget(self.preview_scroll, 1)

        return box

    # ---------- public ----------

    def set_project(self, project: MaterialProject) -> None:
        self.project = project
        self._reset_results()
        self._refresh_readiness()
        self._load_existing_results()
        self._populate_image_combo()

    def refresh(self) -> None:
        self._refresh_readiness()
        self._load_existing_results()
        self._populate_image_combo()

    # ---------- metrics internals ----------

    def _refresh_readiness(self) -> None:
        if self.project is None:
            self.readiness_label.setText("No project loaded.")
            self.run_btn.setEnabled(False)
            self.infer_btn.setEnabled(False)
            return

        amm_ok = amm_available(self.project)
        gmm_ok = gmm_available(self.project)
        val_ok = val_available(self.project)

        self.amm_radio.setEnabled(amm_ok)
        self.gmm_radio.setEnabled(gmm_ok)

        if not (amm_ok or gmm_ok):
            self.readiness_label.setText(
                "No trained models found. Train AMM or GMM first."
            )
            self.readiness_label.setStyleSheet("color: #b8860b;")
            self.run_btn.setEnabled(False)
            self.infer_btn.setEnabled(False)
            return

        if not val_ok:
            self.readiness_label.setText(
                "Val annotation file not found. Run COCO Conversion first. "
                "(Inference preview still works — pick an image below.)"
            )
            self.readiness_label.setStyleSheet("color: #b8860b;")
            self.run_btn.setEnabled(False)
        else:
            bits = []
            if amm_ok:
                bits.append("AMM ✓")
            if gmm_ok:
                bits.append("GMM ✓")
            self.readiness_label.setText(
                f"Ready. {' '.join(bits)} · val annotations found."
            )
            self.readiness_label.setStyleSheet("color: #2e7d32;")
            self.run_btn.setEnabled(not self.runner.is_running())

        if not self.amm_radio.isChecked() and not self.gmm_radio.isChecked():
            if amm_ok:
                self.amm_radio.setChecked(True)
            elif gmm_ok:
                self.gmm_radio.setChecked(True)

        self.infer_btn.setEnabled(
            (amm_ok or gmm_ok) and not self.inference_runner.is_running()
        )

    def _reset_results(self) -> None:
        self.headline_label.setText("")
        self.warning_label.setText("")
        self.per_class_table.setRowCount(0)
        self.cm_table.setRowCount(0)
        self.cm_table.setColumnCount(0)
        self.progress_bar.setValue(0)
        self.stage_label.setText("")
        if hasattr(self, "log_view"):
            self.log_view.clear()
        if hasattr(self, "preview_label"):
            self.preview_label.setText("No preview yet.")
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.adjustSize()
            self.infer_summary_label.setText("")
            self.infer_progress.setValue(0)
            self.infer_stage_label.setText("")

    def _selected_model(self) -> str | None:
        if self.amm_radio.isChecked():
            return "amm"
        if self.gmm_radio.isChecked():
            return "gmm"
        return None

    def _start(self) -> None:
        if self.project is None or self.runner.is_running():
            return
        model = self._selected_model()
        if model is None:
            return
        self._append_log(
            f"=== Evaluating {model.upper()} for project '{self.project.name}' ==="
        )
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.amm_radio.setEnabled(False)
        self.gmm_radio.setEnabled(False)
        self.runner.start(self.project, model)

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
        if total > 0:
            pct = int(round(100 * step / total))
            self.progress_bar.setValue(max(0, min(100, pct)))
        if stage == "done":
            self.progress_bar.setValue(100)
        self.stage_label.setText(f"{stage}: {message}" if message else stage)

    def _on_log(self, line: str) -> None:
        self._append_log(line)

    def _on_finished(self, exit_code: int) -> None:
        self.stop_btn.setEnabled(False)
        if exit_code == 0:
            self._append_log("=== Evaluation complete ===")
            self._load_existing_results()
        else:
            self._append_log(f"=== Evaluation exited with code {exit_code} ===")
        self._refresh_readiness()

    def _append_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)
        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _load_existing_results(self) -> None:
        if self.project is None:
            return
        model = self._selected_model()
        if model is None:
            return
        path = evaluation_path(self.project, model)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except Exception:
            return
        self._render_results(data, model)

    def _render_results(self, data: dict, model: str) -> None:
        accuracy = float(data.get("accuracy", 0.0))
        n = int(data.get("n_instances", 0))
        self.headline_label.setText(
            f"{model.upper()}: accuracy {accuracy:.1%}  ({n} val instances)"
        )

        classes: list[int] = list(data.get("classes", []))
        per_class: dict = data.get("per_class", {})
        cm: list[list[int]] = list(data.get("confusion_matrix", []))

        low = [
            c
            for c in classes
            if per_class.get(str(c), {}).get("support", 0) < LOW_SUPPORT_THRESHOLD
            and per_class.get(str(c), {}).get("support", 0) > 0
        ]
        if low:
            names = ", ".join(self._class_name(c) for c in low)
            self.warning_label.setText(
                f"⚠ Low support for: {names}. Metrics may be noisy with so few instances."
            )
        else:
            self.warning_label.setText("")

        self.per_class_table.setRowCount(len(classes))
        for row, cls in enumerate(classes):
            stats = per_class.get(str(cls), {})
            self.per_class_table.setItem(
                row, 0, QTableWidgetItem(self._class_name(cls))
            )
            self.per_class_table.setItem(
                row, 1, QTableWidgetItem(str(stats.get("support", 0)))
            )
            self.per_class_table.setItem(
                row, 2, QTableWidgetItem(f"{stats.get('precision', 0.0):.3f}")
            )
            self.per_class_table.setItem(
                row, 3, QTableWidgetItem(f"{stats.get('recall', 0.0):.3f}")
            )
            self.per_class_table.setItem(
                row, 4, QTableWidgetItem(f"{stats.get('f1', 0.0):.3f}")
            )

        n_cls = len(classes)
        self.cm_table.setRowCount(n_cls)
        self.cm_table.setColumnCount(n_cls)
        labels = [self._class_name(c) for c in classes]
        self.cm_table.setHorizontalHeaderLabels(labels)
        self.cm_table.setVerticalHeaderLabels(labels)
        max_val = max((max(row) for row in cm), default=0)
        for r, row in enumerate(cm):
            for c, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                if max_val > 0:
                    intensity = val / max_val
                    if r == c:
                        color = QColor(
                            int(255 - intensity * 100),
                            int(255 - intensity * 30),
                            int(255 - intensity * 100),
                        )
                    else:
                        color = QColor(
                            int(255 - intensity * 30),
                            int(255 - intensity * 130),
                            int(255 - intensity * 130),
                        )
                    item.setBackground(color)
                self.cm_table.setItem(r, c, item)

    def _class_name(self, class_id: int) -> str:
        if class_id == 0:
            return "Background"
        if self.project is not None:
            cls = self.project.find_class(class_id)
            if cls is not None:
                return cls.name
        return f"Class {class_id}"

    # ---------- inference internals ----------

    def _populate_image_combo(self) -> None:
        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        if self.project is None:
            self.image_combo.blockSignals(False)
            return

        items: list[tuple[str, str]] = []  # (label, abs path)
        image_root = training_image_root(self.project)

        val_path = val_annotation_path(self.project)
        seen: set[str] = set()
        if val_path.exists():
            try:
                data = json.loads(val_path.read_text())
                for img in data.get("images", []):
                    file_name = img.get("file_name")
                    if not file_name:
                        continue
                    p = image_root / file_name
                    if p.exists() and str(p) not in seen:
                        items.append((f"val · {file_name}", str(p)))
                        seen.add(str(p))
            except Exception:
                pass

        for p in self.project.list_raw_images():
            if str(p) in seen:
                continue
            items.append((p.name, str(p)))
            seen.add(str(p))

        for label, path in items:
            self.image_combo.addItem(label, path)
        self.image_combo.blockSignals(False)

    def _selected_image_path(self) -> Path | None:
        path = self.image_combo.currentData()
        if path:
            return Path(path)
        return None

    def _browse_image(self) -> None:
        if self.project is None:
            return
        start_dir = str(self.project.path)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Pick an image for inference preview",
            start_dir,
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if not path:
            return
        # Add (or select) this path in the combo
        for i in range(self.image_combo.count()):
            if self.image_combo.itemData(i) == path:
                self.image_combo.setCurrentIndex(i)
                return
        self.image_combo.addItem(Path(path).name, path)
        self.image_combo.setCurrentIndex(self.image_combo.count() - 1)

    def _class_info_json(self) -> str:
        if self.project is None:
            return "{}"
        info = {
            str(c.id): {"name": c.name, "color": c.color}
            for c in self.project.classes
        }
        return json.dumps(info)

    def _start_inference(self) -> None:
        if self.project is None or self.inference_runner.is_running():
            return
        model = self._selected_model()
        if model is None:
            self._append_log("[inference] Pick a model (AMM or GMM) first.")
            return
        image_path = self._selected_image_path()
        if image_path is None or not image_path.exists():
            self._append_log("[inference] Pick a valid image first.")
            return
        self.infer_btn.setEnabled(False)
        self.infer_stop_btn.setEnabled(True)
        self.infer_progress.setValue(0)
        self.infer_summary_label.setText("")
        self.preview_label.setText("Running…")
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.adjustSize()
        self._append_log(
            f"=== Inference {model.upper()} on {image_path.name} ==="
        )
        self.inference_runner.start(
            self.project,
            model,
            image_path,
            size_threshold=self.size_threshold.value(),
            class_info_json=self._class_info_json(),
        )

    def _stop_inference(self) -> None:
        if not self.inference_runner.is_running():
            return
        self._append_log("[stopping inference…]")
        self.inference_runner.stop()

    def _on_inference_progress(self, payload: dict) -> None:
        step = int(payload.get("step", 0))
        total = int(payload.get("total", 0))
        stage = str(payload.get("stage", ""))
        message = str(payload.get("message", ""))
        if total > 0:
            pct = int(round(100 * step / total))
            self.infer_progress.setValue(max(0, min(100, pct)))
        if stage == "done":
            self.infer_progress.setValue(100)
        self.infer_stage_label.setText(
            f"{stage}: {message}" if message else stage
        )

    def _on_inference_finished(
        self, exit_code: int, image_path: str, json_path: str
    ) -> None:
        self.infer_stop_btn.setEnabled(False)
        if exit_code == 0 and image_path and Path(image_path).exists():
            pix = QPixmap(image_path)
            if not pix.isNull():
                self.preview_label.setPixmap(pix)
                self.preview_label.resize(pix.size())
            else:
                self.preview_label.setText("Could not load preview image.")
            n = 0
            try:
                if json_path and Path(json_path).exists():
                    data = json.loads(Path(json_path).read_text())
                    n = int(data.get("n_detections", 0))
            except Exception:
                pass
            self.infer_summary_label.setText(
                f"{n} detection{'s' if n != 1 else ''} · "
                f"saved to {inference_output_dir(self.project).name}/"
            )
            self._append_log("=== Inference complete ===")
        else:
            self.preview_label.setText("Inference failed — see log below.")
            self.infer_summary_label.setText("")
            self._append_log(
                f"=== Inference exited with code {exit_code} ==="
            )
        self._refresh_readiness()
