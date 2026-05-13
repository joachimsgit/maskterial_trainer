from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..pipeline.project import IMAGE_EXTS, MaterialProject


class _DropListWidget(QListWidget):
    files_dropped = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setDragEnabled(False)
        self.setDragDropMode(QAbstractItemView.NoDragDrop)
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(QSize(160, 160))
        self.setResizeMode(QListWidget.Adjust)
        self.setSpacing(8)
        self.setMovement(QListWidget.Static)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = [
                Path(u.toLocalFile())
                for u in event.mimeData().urls()
                if u.isLocalFile()
            ]
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class ImagesView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.project: MaterialProject | None = None

        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.add_btn = QPushButton("Add images…")
        self.add_btn.clicked.connect(self._on_add)
        toolbar.addWidget(self.add_btn)
        toolbar.addStretch(1)
        self.count_label = QLabel("0 images")
        toolbar.addWidget(self.count_label)
        layout.addLayout(toolbar)

        hint = QLabel(
            "Drop image files (or folders) anywhere in the grid below to import."
        )
        hint.setStyleSheet("color: #888;")
        layout.addWidget(hint)

        # Flatfield row
        flatfield_row = QHBoxLayout()
        flatfield_row.addWidget(QLabel("Flatfield:"))
        self.flatfield_status = QLabel("(none)")
        self.flatfield_status.setStyleSheet("color: #555;")
        flatfield_row.addWidget(self.flatfield_status, 1)
        self.set_flatfield_btn = QPushButton("Set…")
        self.set_flatfield_btn.clicked.connect(self._on_set_flatfield)
        self.clear_flatfield_btn = QPushButton("Clear")
        self.clear_flatfield_btn.clicked.connect(self._on_clear_flatfield)
        self.clear_flatfield_btn.setEnabled(False)
        self.flatfield_enable_check = QCheckBox(
            "Apply during COCO Conversion / training"
        )
        self.flatfield_enable_check.setEnabled(False)
        self.flatfield_enable_check.toggled.connect(self._on_toggle_flatfield)
        flatfield_row.addWidget(self.set_flatfield_btn)
        flatfield_row.addWidget(self.clear_flatfield_btn)
        flatfield_row.addWidget(self.flatfield_enable_check)
        layout.addLayout(flatfield_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setStyleSheet("color: #ddd;")
        layout.addWidget(sep)

        self.grid = _DropListWidget()
        self.grid.files_dropped.connect(self._import_paths)
        layout.addWidget(self.grid, 1)

    def set_project(self, project: MaterialProject) -> None:
        self.project = project
        self._refresh_flatfield()
        self.refresh()

    def refresh(self) -> None:
        self.grid.clear()
        if self.project is None:
            self.count_label.setText("0 images")
            return
        images = self.project.list_raw_images()
        for path in images:
            pix = QPixmap(str(path))
            if pix.isNull():
                continue
            icon = QIcon(
                pix.scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            item = QListWidgetItem(icon, path.name)
            item.setData(Qt.UserRole, str(path))
            self.grid.addItem(item)
        self.count_label.setText(
            f"{len(images)} image{'s' if len(images) != 1 else ''}"
        )

    def _refresh_flatfield(self) -> None:
        if self.project is None:
            self.flatfield_status.setText("(none)")
            self.set_flatfield_btn.setEnabled(False)
            self.clear_flatfield_btn.setEnabled(False)
            self.flatfield_enable_check.setEnabled(False)
            return
        self.set_flatfield_btn.setEnabled(True)
        if self.project.flatfield_path:
            self.flatfield_status.setText(self.project.flatfield_path)
            self.clear_flatfield_btn.setEnabled(True)
            self.flatfield_enable_check.setEnabled(True)
        else:
            self.flatfield_status.setText("(none)")
            self.clear_flatfield_btn.setEnabled(False)
            self.flatfield_enable_check.setEnabled(False)
        self.flatfield_enable_check.blockSignals(True)
        self.flatfield_enable_check.setChecked(self.project.flatfield_enabled)
        self.flatfield_enable_check.blockSignals(False)

    def _on_set_flatfield(self) -> None:
        if self.project is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose flatfield image",
            str(self.project.path),
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if not path:
            return
        try:
            self.project.set_flatfield(Path(path))
        except Exception as e:
            QMessageBox.warning(self, "Could not set flatfield", str(e))
            return
        self._refresh_flatfield()
        self.refresh()

    def _on_clear_flatfield(self) -> None:
        if self.project is None:
            return
        self.project.clear_flatfield()
        self._refresh_flatfield()
        self.refresh()

    def _on_toggle_flatfield(self, checked: bool) -> None:
        if self.project is None:
            return
        self.project.set_flatfield_enabled(checked)

    def _on_add(self) -> None:
        if self.project is None:
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add images",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        self._import_paths([Path(p) for p in paths])

    def _import_paths(self, paths: list[Path]) -> None:
        if self.project is None or not paths:
            return
        errors: list[str] = []
        for src in paths:
            try:
                if src.is_dir():
                    for child in sorted(src.iterdir()):
                        if (
                            child.is_file()
                            and child.suffix.lower() in IMAGE_EXTS
                        ):
                            self.project.import_image(child)
                elif src.suffix.lower() in IMAGE_EXTS:
                    self.project.import_image(src)
                else:
                    errors.append(f"{src.name}: unsupported file type")
            except Exception as e:
                errors.append(f"{src.name}: {e}")
        self.refresh()
        if errors:
            QMessageBox.warning(
                self,
                "Some images couldn't be imported",
                "\n".join(errors[:10]) + ("\n…" if len(errors) > 10 else ""),
            )
