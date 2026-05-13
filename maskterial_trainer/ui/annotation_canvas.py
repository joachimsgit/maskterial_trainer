from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRect, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from ..pipeline.annotation_io import load_instance_mask, save_instance_mask

PALETTE = [
    (220, 30, 30), (30, 180, 30), (30, 30, 220),
    (220, 180, 30), (220, 30, 180), (30, 180, 220),
    (180, 30, 180), (180, 180, 30), (30, 180, 180),
    (220, 120, 30), (30, 220, 120), (120, 30, 220),
]
MARKER_RADIUS = 4
OVERLAY_ALPHA = 130
MIN_ZOOM = 0.1
MAX_ZOOM = 40.0
WHEEL_FACTOR = 1.2
PAN_THRESHOLD_PX = 4


def _label_color(label: int) -> tuple[int, int, int]:
    if label <= 0:
        return (0, 0, 0)
    return PALETTE[(label - 1) % len(PALETTE)]


class AnnotationCanvas(QWidget):
    mode_changed = Signal(str)
    dirty_changed = Signal(bool)
    saved = Signal()
    status = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(400, 400)

        self._image: Optional[np.ndarray] = None
        self._image_path: Optional[Path] = None
        self._mask_path: Optional[Path] = None

        self._base_mask: Optional[np.ndarray] = None
        self._markers: Optional[np.ndarray] = None
        self._watershed_result: Optional[np.ndarray] = None
        self._polygons: list[list[tuple[int, int]]] = []
        self._poly_in_progress: list[tuple[int, int]] = []
        self._next_fg_label: int = 2

        self._mode: str = "watershed"
        self._show_mask: bool = True
        self._dirty: bool = False
        self._undo_stack: list[dict] = []

        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0

        self._lmb_down_pos: Optional[tuple[float, float]] = None
        self._lmb_pan_anchor: tuple[float, float] = (0.0, 0.0)
        self._lmb_panning: bool = False

    # ---------- public API ----------

    def load_image(self, image_path: Path, mask_path: Path) -> None:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            self.status.emit(f"Failed to read {image_path.name}")
            self._image = None
            self.update()
            return
        self._image = img
        self._image_path = image_path
        self._mask_path = mask_path

        h, w = img.shape[:2]
        self._markers = np.zeros((h, w), dtype=np.int32)
        self._watershed_result = None
        self._polygons = []
        self._poly_in_progress = []
        self._undo_stack = []

        loaded = load_instance_mask(mask_path)
        if loaded is not None and loaded.shape == (h, w):
            self._base_mask = loaded
            self._next_fg_label = max(int(loaded.max()), 1) + 2
            self.status.emit(f"Loaded existing mask ({int(loaded.max())} instances)")
        else:
            self._base_mask = None
            self._next_fg_label = 2

        self._reset_view_state()
        self._lmb_down_pos = None
        self._lmb_panning = False
        self.unsetCursor()
        self._set_dirty(False)
        self.update()

    def save_mask(self) -> bool:
        if self._mask_path is None or self._image is None:
            return False
        composite = self._composite_mask()
        if composite.max() == 0:
            if self._mask_path.exists():
                self._mask_path.unlink()
            self._set_dirty(False)
            self.saved.emit()
            self.status.emit(f"Empty mask — removed {self._mask_path.name}")
            return True
        save_instance_mask(self._mask_path, composite)
        self._set_dirty(False)
        self.saved.emit()
        self.status.emit(
            f"Saved {self._mask_path.name} ({int(composite.max())} instances)"
        )
        return True

    def set_mode(self, mode: str) -> None:
        if mode == self._mode or mode not in {"watershed", "polygon"}:
            return
        self._poly_in_progress = []
        self._mode = mode
        self.mode_changed.emit(mode)
        self.update()

    def get_mode(self) -> str:
        return self._mode

    def toggle_mask_visibility(self) -> None:
        self._show_mask = not self._show_mask
        self.update()

    def fit_to_window(self) -> None:
        self._reset_view_state()
        self.update()

    def is_dirty(self) -> bool:
        return self._dirty

    def has_image(self) -> bool:
        return self._image is not None

    def undo(self) -> None:
        if not self._undo_stack:
            return
        snap = self._undo_stack.pop()
        self._markers = snap["markers"]
        self._watershed_result = snap["watershed"]
        self._polygons = snap["polygons"]
        self._next_fg_label = snap["next_label"]
        self._set_dirty(True)
        self.update()

    def clear_edits(self) -> None:
        if self._image is None:
            return
        self._snap_undo()
        h, w = self._image.shape[:2]
        self._markers = np.zeros((h, w), dtype=np.int32)
        self._watershed_result = None
        self._polygons = []
        self._poly_in_progress = []
        self._base_mask = None
        self._next_fg_label = 2
        self._set_dirty(True)
        self.update()

    # ---------- mouse + wheel + keys ----------

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if self._image is None:
            return
        if e.button() == Qt.LeftButton:
            self._lmb_down_pos = (e.position().x(), e.position().y())
            self._lmb_pan_anchor = (self._pan_x, self._pan_y)
            self._lmb_panning = False
            return
        if e.button() == Qt.RightButton:
            ix, iy = self._widget_to_image(e.position().x(), e.position().y())
            if not self._in_bounds(ix, iy):
                return
            self._handle_rmb_click(ix, iy)

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self._image is None or self._lmb_down_pos is None:
            return
        dx = e.position().x() - self._lmb_down_pos[0]
        dy = e.position().y() - self._lmb_down_pos[1]
        if not self._lmb_panning:
            if abs(dx) + abs(dy) > PAN_THRESHOLD_PX:
                self._lmb_panning = True
                self.setCursor(Qt.ClosedHandCursor)
        if self._lmb_panning:
            self._pan_x = self._lmb_pan_anchor[0] + dx
            self._pan_y = self._lmb_pan_anchor[1] + dy
            self.update()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if e.button() != Qt.LeftButton or self._lmb_down_pos is None:
            return
        was_panning = self._lmb_panning
        click_pos = self._lmb_down_pos
        self._lmb_panning = False
        self._lmb_down_pos = None
        self.unsetCursor()
        if was_panning:
            return
        ix, iy = self._widget_to_image(click_pos[0], click_pos[1])
        if not self._in_bounds(ix, iy):
            return
        self._handle_lmb_click(ix, iy)

    def wheelEvent(self, e: QWheelEvent) -> None:
        if self._image is None:
            return
        delta = e.angleDelta().y()
        if delta == 0:
            return
        factor = WHEEL_FACTOR if delta > 0 else 1.0 / WHEEL_FACTOR
        pos = e.position()
        self._zoom_at(pos.x(), pos.y(), factor)
        e.accept()

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self._mode == "polygon" and len(self._poly_in_progress) >= 3:
                self._snap_undo()
                self._polygons.append(self._poly_in_progress)
                self._poly_in_progress = []
                self._set_dirty(True)
                self.update()
        elif e.key() == Qt.Key_Escape:
            if self._poly_in_progress:
                self._poly_in_progress = []
                self.update()
        else:
            super().keyPressEvent(e)

    # ---------- click dispatch ----------

    def _handle_lmb_click(self, ix: int, iy: int) -> None:
        if self._mode == "watershed":
            self._snap_undo()
            self._stroke_label = self._next_fg_label
            self._next_fg_label += 1
            self._paint_marker(ix, iy, self._stroke_label)
            self._set_dirty(True)
            self._run_watershed()
            self.update()
        elif self._mode == "polygon":
            self._poly_in_progress.append((ix, iy))
            self.update()

    def _handle_rmb_click(self, ix: int, iy: int) -> None:
        if self._mode == "watershed":
            self._snap_undo()
            self._paint_marker(ix, iy, 1)
            self._set_dirty(True)
            self._run_watershed()
            self.update()
        elif self._mode == "polygon":
            if len(self._poly_in_progress) >= 3:
                self._snap_undo()
                self._polygons.append(self._poly_in_progress)
                self._poly_in_progress = []
                self._set_dirty(True)
            else:
                self._poly_in_progress = []
            self.update()

    # ---------- painting ----------

    def paintEvent(self, _e) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        if self._image is None:
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image loaded")
            return

        ox, oy, scale = self._transform()
        h, w = self._image.shape[:2]
        target = QRect(int(ox), int(oy), int(round(w * scale)), int(round(h * scale)))

        rgb = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        painter.drawImage(target, qimg)

        if self._show_mask:
            composite = self._composite_mask()
            if composite.max() > 0:
                rgba = self._colorize_mask(composite)
                qmask = QImage(
                    rgba.data, w, h, rgba.strides[0], QImage.Format_RGBA8888
                ).copy()
                painter.drawImage(target, qmask)

        if self._mode == "watershed" and self._markers is not None:
            mo = self._colorize_markers(self._markers)
            qm = QImage(mo.data, w, h, mo.strides[0], QImage.Format_RGBA8888).copy()
            painter.drawImage(target, qm)

        if self._poly_in_progress:
            painter.setPen(QPen(QColor(255, 230, 0), 2))
            painter.setBrush(QColor(255, 230, 0, 60))
            poly_widget = [
                QPointF(ox + p[0] * scale, oy + p[1] * scale)
                for p in self._poly_in_progress
            ]
            for i in range(len(poly_widget) - 1):
                painter.drawLine(poly_widget[i], poly_widget[i + 1])
            for p in poly_widget:
                painter.drawEllipse(p, 4, 4)

    # ---------- view transform ----------

    def _reset_view_state(self) -> None:
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0

    def _transform(self) -> tuple[float, float, float]:
        cw, ch = self.width(), self.height()
        h, w = self._image.shape[:2]
        base = min(cw / w, ch / h)
        scale = base * self._zoom
        ox = (cw - w * scale) / 2 + self._pan_x
        oy = (ch - h * scale) / 2 + self._pan_y
        return ox, oy, scale

    def _widget_to_image(self, x: float, y: float) -> tuple[int, int]:
        ox, oy, scale = self._transform()
        return int((x - ox) / scale), int((y - oy) / scale)

    def _widget_to_image_float(self, x: float, y: float) -> tuple[float, float]:
        ox, oy, scale = self._transform()
        return (x - ox) / scale, (y - oy) / scale

    def _in_bounds(self, ix: int, iy: int) -> bool:
        if self._image is None:
            return False
        h, w = self._image.shape[:2]
        return 0 <= ix < w and 0 <= iy < h

    def _zoom_at(self, cx: float, cy: float, factor: float) -> None:
        ix, iy = self._widget_to_image_float(cx, cy)
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self._zoom * factor))
        if new_zoom == self._zoom:
            return
        self._zoom = new_zoom
        cw, ch = self.width(), self.height()
        h, w = self._image.shape[:2]
        base = min(cw / w, ch / h)
        scale = base * self._zoom
        self._pan_x = cx - (cw - w * scale) / 2 - ix * scale
        self._pan_y = cy - (ch - h * scale) / 2 - iy * scale
        self.update()

    # ---------- internals ----------

    def _paint_marker(self, ix: int, iy: int, label: int) -> None:
        cv2.circle(self._markers, (ix, iy), MARKER_RADIUS, int(label), -1)

    def _run_watershed(self) -> None:
        if self._image is None or self._markers is None:
            return
        unique = np.unique(self._markers[self._markers > 0])
        if len(unique) < 2:
            self.status.emit(
                "Add both foreground (left-click) and background (right-click) "
                "markers — watershed needs both."
            )
            self._watershed_result = None
            return
        result = self._markers.copy()
        cv2.watershed(self._image, result)
        self._watershed_result = result

    def _composite_mask(self) -> np.ndarray:
        h, w = self._image.shape[:2]
        if self._base_mask is not None:
            out = self._base_mask.copy().astype(np.int32)
        else:
            out = np.zeros((h, w), dtype=np.int32)
        next_label = int(out.max()) + 1 if out.max() > 0 else 1

        if self._watershed_result is not None:
            ws = self._watershed_result
            ws_labels = sorted(int(x) for x in np.unique(ws) if x >= 2)
            for ws_label in ws_labels:
                pixels = (ws == ws_label) & (out == 0)
                if pixels.any():
                    out[pixels] = next_label
                    next_label += 1

        for poly in self._polygons:
            pts = np.array(poly, dtype=np.int32)
            tmp = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(tmp, [pts], 1)
            assign = (tmp == 1) & (out == 0)
            if assign.any():
                out[assign] = next_label
                next_label += 1
        return out

    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        out = np.zeros((h, w, 4), dtype=np.uint8)
        for label in np.unique(mask):
            if label <= 0:
                continue
            color = _label_color(int(label))
            out[mask == label] = (*color, OVERLAY_ALPHA)
        return np.ascontiguousarray(out)

    def _colorize_markers(self, markers: np.ndarray) -> np.ndarray:
        h, w = markers.shape
        out = np.zeros((h, w, 4), dtype=np.uint8)
        out[markers == 1] = (255, 255, 255, 220)
        out[markers >= 2] = (0, 230, 230, 220)
        return np.ascontiguousarray(out)

    def _snap_undo(self) -> None:
        self._undo_stack.append(
            {
                "markers": self._markers.copy() if self._markers is not None else None,
                "watershed": (
                    self._watershed_result.copy()
                    if self._watershed_result is not None
                    else None
                ),
                "polygons": [list(p) for p in self._polygons],
                "next_label": self._next_fg_label,
            }
        )
        if len(self._undo_stack) > 30:
            self._undo_stack.pop(0)

    def _set_dirty(self, value: bool) -> None:
        if self._dirty != value:
            self._dirty = value
            self.dirty_changed.emit(value)
