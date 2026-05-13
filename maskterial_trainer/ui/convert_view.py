from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from ..pipeline.project import MaterialProject

EXPLAINER = (
    "MaskTerial's training pipeline expects the dataset in COCO format with "
    "RLE-encoded instance masks. Pressing the button below will:\n\n"
    "  • Split your annotated images 80:20 into train and validation sets "
    "(persisted in splits/split.json so the split is stable across re-runs)\n"
    "  • Convert each instance mask to a COCO annotation with an RLE-encoded mask\n"
    "  • Filter instances smaller than 300 pixels (matches the MaskTerial "
    "training convention)\n"
    "  • Generate four files in coco/:\n"
    "       – train_annotations.json (for segmentation training)\n"
    "       – train_annotations_with_class.json (for classification training)\n"
    "       – val_annotations.json\n"
    "       – val_annotations_with_class.json\n\n"
    "You only need to run this whenever your annotations change."
)


class ConvertView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.project: MaterialProject | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 48, 48, 48)

        title = QLabel("COCO Conversion")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        explainer = QLabel(EXPLAINER)
        explainer.setWordWrap(True)
        explainer.setStyleSheet("color: #555;")
        layout.addWidget(explainer)
        layout.addSpacing(16)

        self.convert_btn = QPushButton("Convert to COCO")
        self.convert_btn.clicked.connect(self._convert)
        layout.addWidget(self.convert_btn, alignment=Qt.AlignLeft)
        layout.addSpacing(8)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("padding: 12px; color: #333;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    def set_project(self, project: MaterialProject) -> None:
        self.project = project
        self.status_label.setText("")

    def _convert(self) -> None:
        if self.project is None:
            return
        self.convert_btn.setEnabled(False)
        self.status_label.setText("Converting…")
        self.repaint()
        try:
            from ..pipeline.coco_export import export_coco

            result = export_coco(
                self.project,
                progress=lambda i, n, name: self._on_progress(i, n, name),
            )
        except ImportError as e:
            self.status_label.setText(
                f"Missing dependency: {e}. Install pycocotools "
                "(pip install pycocotools)."
            )
            self.convert_btn.setEnabled(True)
            return
        except Exception as e:
            self.status_label.setText(f"Error during conversion: {e}")
            self.convert_btn.setEnabled(True)
            return
        finally:
            self.convert_btn.setEnabled(True)

        if "error" in result:
            self.status_label.setText(result["error"])
            return

        ff_note = (
            " · flatfield correction applied to staged images."
            if result.get("flatfield_applied")
            else ""
        )
        self.status_label.setText(
            f"✓ Converted {result['n_images']} image(s) "
            f"({result['n_train']} train · {result['n_val']} val), "
            f"{result['n_instances']} instance(s) total, "
            f"{result['n_class_instances']} with class assignment.{ff_note}\n"
            f"Files saved in {result['coco_dir']}.\n"
            f"Staged images for training in {result.get('image_root', '?')}.\n"
            f"Split persisted in {result['split_path']}."
        )

    def _on_progress(self, i: int, n: int, name: str) -> None:
        if n == 0:
            return
        if i >= n:
            self.status_label.setText("Writing COCO files…")
        else:
            self.status_label.setText(f"Processing {i + 1}/{n}: {name}")
        self.repaint()
