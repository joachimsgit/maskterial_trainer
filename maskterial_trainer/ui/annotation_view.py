from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..pipeline.annotation_io import mask_path_for_image
from ..pipeline.project import MaterialProject
from .annotation_canvas import AnnotationCanvas


class AnnotationView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.project: MaterialProject | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        left = QWidget()
        left.setFixedWidth(220)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.addWidget(QLabel("Images"))
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._on_image_row_changed)
        ll.addWidget(self.image_list, 1)
        layout.addWidget(left)

        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(8, 8, 8, 0)
        self.watershed_btn = QPushButton("Watershed (W)")
        self.watershed_btn.setCheckable(True)
        self.watershed_btn.setChecked(True)
        self.watershed_btn.clicked.connect(lambda: self._set_mode("watershed"))
        self.polygon_btn = QPushButton("Polygon (P)")
        self.polygon_btn.setCheckable(True)
        self.polygon_btn.clicked.connect(lambda: self._set_mode("polygon"))
        self.save_btn = QPushButton("Save (S)")
        self.save_btn.clicked.connect(self._save_current)
        self.toggle_mask_btn = QPushButton("Toggle mask (M)")
        self.toggle_mask_btn.clicked.connect(self._toggle_mask)
        self.undo_btn = QPushButton("Undo (Z)")
        self.undo_btn.clicked.connect(self._undo)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_edits)
        toolbar.addWidget(self.watershed_btn)
        toolbar.addWidget(self.polygon_btn)
        toolbar.addSpacing(16)
        toolbar.addWidget(self.save_btn)
        toolbar.addWidget(self.toggle_mask_btn)
        toolbar.addWidget(self.undo_btn)
        toolbar.addWidget(self.clear_btn)
        toolbar.addStretch(1)
        cl.addLayout(toolbar)

        self.canvas = AnnotationCanvas()
        self.canvas.dirty_changed.connect(self._on_dirty_changed)
        self.canvas.status.connect(self._set_status)
        self.canvas.saved.connect(lambda: None)
        cl.addWidget(self.canvas, 1)

        self.status_label = QLabel(self._default_status("watershed"))
        self.status_label.setStyleSheet("color: #555; padding: 6px;")
        self.status_label.setWordWrap(True)
        cl.addWidget(self.status_label)

        layout.addWidget(center, 1)

        for key, slot in (
            (Qt.Key_A, self._prev_image),
            (Qt.Key_D, self._next_image),
            (Qt.Key_M, self._toggle_mask),
            (Qt.Key_S, self._save_current),
            (Qt.Key_W, lambda: self._set_mode("watershed")),
            (Qt.Key_P, lambda: self._set_mode("polygon")),
            (Qt.Key_Z, self._undo),
            (Qt.Key_F, self._fit_to_window),
        ):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)

    def set_project(self, project: MaterialProject) -> None:
        self.project = project
        self.refresh()

    def refresh(self) -> None:
        current_name = (
            self.image_list.currentItem().text()
            if self.image_list.currentItem() is not None
            else None
        )
        self.image_list.blockSignals(True)
        self.image_list.clear()
        if self.project is not None:
            for path in self.project.list_raw_images():
                item = QListWidgetItem(path.name)
                item.setData(Qt.UserRole, str(path))
                self.image_list.addItem(item)
        self.image_list.blockSignals(False)
        if current_name:
            for i in range(self.image_list.count()):
                if self.image_list.item(i).text() == current_name:
                    self.image_list.setCurrentRow(i)
                    return
        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)

    # ---------- internals ----------

    def _save_current(self) -> bool:
        return self.canvas.save_mask()

    def _on_image_row_changed(self, row: int) -> None:
        if row < 0 or self.project is None:
            return
        if self.canvas.is_dirty():
            self.canvas.save_mask()
        item = self.image_list.item(row)
        image_path = Path(item.data(Qt.UserRole))
        mask_path = mask_path_for_image(self.project.path, image_path)
        self.canvas.load_image(image_path, mask_path)
        self.canvas.setFocus()

    def _prev_image(self) -> None:
        row = self.image_list.currentRow()
        if row > 0:
            self.image_list.setCurrentRow(row - 1)

    def _next_image(self) -> None:
        row = self.image_list.currentRow()
        if 0 <= row < self.image_list.count() - 1:
            self.image_list.setCurrentRow(row + 1)

    def _toggle_mask(self) -> None:
        self.canvas.toggle_mask_visibility()

    def _undo(self) -> None:
        self.canvas.undo()

    def _fit_to_window(self) -> None:
        self.canvas.fit_to_window()

    def _clear_edits(self) -> None:
        if not self.canvas.has_image():
            return
        ans = QMessageBox.question(
            self,
            "Clear all edits?",
            "This wipes watershed markers, polygons, and any loaded mask "
            "for this image. Continue?",
        )
        if ans == QMessageBox.Yes:
            self.canvas.clear_edits()

    def _set_mode(self, mode: str) -> None:
        self.canvas.set_mode(mode)
        self.watershed_btn.setChecked(mode == "watershed")
        self.polygon_btn.setChecked(mode == "polygon")
        self._set_status(self._default_status(mode))
        self.canvas.setFocus()

    def _on_dirty_changed(self, dirty: bool) -> None:
        if dirty:
            self._set_status("Unsaved changes — press S to save.")

    def _set_status(self, msg: str) -> None:
        self.status_label.setText(msg)

    @staticmethod
    def _default_status(mode: str) -> str:
        nav = "Wheel zooms at cursor · LMB-drag pans · F resets view · A/D nav · M toggle mask · S save."
        if mode == "watershed":
            return f"Watershed: left-click flake markers, right-click background.  {nav}"
        return (
            f"Polygon: left-click vertices, right-click or Enter to close, ESC cancels.  {nav}"
        )
