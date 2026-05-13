from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..pipeline.export_bundle import (
    default_export_dir,
    export_project,
    has_amm,
    has_amm_eval,
    has_gmm,
    has_gmm_eval,
    has_segmentation,
)
from ..pipeline.project import MaterialProject


class ExportView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.project: MaterialProject | None = None
        self._last_dest: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Export bundle")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        explainer = QLabel(
            "Packages your annotated images, masks, and trained models into a "
            "single folder ready to upload to the inference website. Each model "
            "gets its own subfolder containing exactly the files the upload "
            "endpoint expects."
        )
        explainer.setWordWrap(True)
        explainer.setStyleSheet("color: #555;")
        layout.addWidget(explainer)
        layout.addSpacing(8)

        # Status pane
        layout.addWidget(QLabel("What will be bundled"))
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(
            "padding: 8px; background: #f6f6f6; border: 1px solid #ddd;"
        )
        self.status_label.setTextFormat(Qt.RichText)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addSpacing(8)

        # Destination row
        dest_row = QHBoxLayout()
        dest_row.addWidget(QLabel("Output folder:"))
        self.dest_edit = QLineEdit()
        dest_row.addWidget(self.dest_edit, 1)
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._browse)
        dest_row.addWidget(self.browse_btn)
        layout.addLayout(dest_row)

        self.zip_check = QCheckBox("Also write a .zip archive next to the folder")
        self.zip_check.setChecked(True)
        layout.addWidget(self.zip_check)

        # Buttons
        btn_row = QHBoxLayout()
        self.export_btn = QPushButton("Export bundle")
        self.export_btn.clicked.connect(self._export)
        self.open_btn = QPushButton("Open output folder")
        self.open_btn.clicked.connect(self._open_output)
        self.open_btn.setEnabled(False)
        btn_row.addWidget(self.export_btn)
        btn_row.addWidget(self.open_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        # Log
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(180)
        self.log_view.setStyleSheet(
            "QPlainTextEdit { background: #1e1e1e; color: #d0d0d0; "
            "font-family: Consolas, monospace; font-size: 11px; }"
        )
        layout.addWidget(self.log_view, 1)

    # ---------- public ----------

    def set_project(self, project: MaterialProject) -> None:
        self.project = project
        self.dest_edit.setText(str(default_export_dir(project)))
        self._last_dest = None
        self.open_btn.setEnabled(False)
        self.log_view.clear()
        self.refresh()

    def refresh(self) -> None:
        if self.project is None:
            self.status_label.setText("No project loaded.")
            self.export_btn.setEnabled(False)
            return

        n_images = len(self.project.list_raw_images())
        instance_dir = self.project.path / "instance_masks"
        semantic_dir = self.project.path / "semantic_masks"
        n_inst = (
            sum(1 for f in instance_dir.iterdir() if f.suffix.lower() == ".png")
            if instance_dir.exists()
            else 0
        )
        n_sem = (
            sum(1 for f in semantic_dir.iterdir() if f.suffix.lower() == ".png")
            if semantic_dir.exists()
            else 0
        )

        amm_state = self._model_state(has_amm(self.project), has_amm_eval(self.project))
        gmm_state = self._model_state(has_gmm(self.project), has_gmm_eval(self.project))
        seg_state = self._model_state(has_segmentation(self.project), False)

        rows = [
            f"<b>Images:</b> {n_images}",
            f"<b>Instance masks:</b> {n_inst}",
            f"<b>Semantic masks:</b> {n_sem}",
            f"<b>AMM:</b> {amm_state}",
            f"<b>GMM:</b> {gmm_state}",
            f"<b>Segmentation:</b> {seg_state}",
        ]
        self.status_label.setText("<br>".join(rows))

        self.export_btn.setEnabled(n_images > 0)

    # ---------- internals ----------

    @staticmethod
    def _model_state(present: bool, evaluated: bool) -> str:
        if not present:
            return "<span style='color:#999;'>not trained</span>"
        if evaluated:
            return "<span style='color:#2e7d32;'>✓ trained · evaluated</span>"
        return "<span style='color:#2e7d32;'>✓ trained</span>"

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Choose output folder",
            self.dest_edit.text() or str(Path.home()),
        )
        if path:
            self.dest_edit.setText(path)

    def _export(self) -> None:
        if self.project is None:
            return
        raw_dest = self.dest_edit.text().strip()
        if not raw_dest:
            QMessageBox.warning(self, "No output folder", "Pick a folder first.")
            return
        dest = Path(raw_dest).resolve()

        # Refuse to wipe paths that are too dangerous to delete contents of —
        # the user probably mistyped, and the cost of a wrong delete is high.
        forbidden_roots = {
            Path.home().resolve(),
            self.project.path.resolve(),
            Path(dest.anchor).resolve() if dest.anchor else None,
        }
        if dest in forbidden_roots:
            QMessageBox.warning(
                self,
                "Refusing to overwrite",
                f"Refusing to use {dest} as the export folder — it's a "
                "drive root, your home folder, or the project folder. "
                "Pick a dedicated output folder.",
            )
            return

        if dest.exists() and any(dest.iterdir()):
            children = list(dest.iterdir())
            # Looks-like-a-previous-export check: at least project.json + README.md
            # would have been written by export_project last time. If neither is
            # there, we treat the folder as foreign and refuse to wipe.
            looks_like_export = (
                (dest / "project.json").exists() and (dest / "README.md").exists()
            )
            if not looks_like_export:
                QMessageBox.warning(
                    self,
                    "Folder is not empty",
                    f"{dest} contains {len(children)} item(s) and does not look "
                    "like a previous MaskTerial export "
                    "(missing project.json and README.md). Pick an empty folder "
                    "or a folder that this app previously exported into.",
                )
                return

            preview = ", ".join(c.name for c in children[:5])
            if len(children) > 5:
                preview += f", … (+{len(children) - 5} more)"
            ans = QMessageBox.question(
                self,
                "Overwrite previous export?",
                f"This will delete the existing contents of:\n\n  {dest}\n\n"
                f"Contains: {preview}\n\nContinue?",
            )
            if ans != QMessageBox.Yes:
                return
            import shutil as _shutil
            for child in children:
                if child.is_dir():
                    _shutil.rmtree(child)
                else:
                    child.unlink()

        self.export_btn.setEnabled(False)
        self.open_btn.setEnabled(False)
        self.log_view.clear()
        self._append_log(f"Exporting to {dest}…")
        self.repaint()

        try:
            result = export_project(
                self.project,
                dest,
                include_zip=self.zip_check.isChecked(),
                progress=self._append_log,
            )
        except Exception as e:
            self._append_log(f"Export failed: {e}")
            self.export_btn.setEnabled(True)
            return

        self.export_btn.setEnabled(True)
        self._last_dest = Path(result["dest"])
        self.open_btn.setEnabled(True)
        self._append_log("--- Summary ---")
        self._append_log(f"Output folder: {result['dest']}")
        if result["zip"]:
            self._append_log(f"Zip archive:   {result['zip']}")
        self._append_log(
            f"Images: {result['n_images']} · "
            f"instance masks: {result['n_instance_masks']} · "
            f"semantic masks: {result['n_semantic_masks']}"
        )
        if result["bundled_models"]:
            self._append_log(
                "Bundled models: " + ", ".join(result["bundled_models"])
            )
        else:
            self._append_log("No trained models were bundled.")

    def _open_output(self) -> None:
        if self._last_dest is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_dest)))

    def _append_log(self, line: str) -> None:
        self.log_view.appendPlainText(line)
        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())
