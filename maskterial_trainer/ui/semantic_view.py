from __future__ import annotations

from collections import OrderedDict, defaultdict

import cv2
import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QCursor, QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..pipeline.annotation_io import load_instance_mask, mask_path_for_image
from ..pipeline.contrasts import InstanceContrast, aggregate_project_contrasts
from ..pipeline.flatfield import load_flatfield, remove_vignette
from ..pipeline.project import MaterialProject
from ..pipeline.semantic_io import (
    load_semantic_mask,
    save_semantic_mask,
    semantic_mask_path_for_image,
)

PREVIEW_PADDING_PX = 30
PREVIEW_MAX_SIDE = 512
IMAGE_CACHE_SIZE = 4

UNASSIGNED_LABEL = "(Unassigned)"
UNASSIGNED_COLOR = "#888888"

CLASS_PALETTE = [
    "#e6194b",  # red
    "#4363d8",  # blue
    "#3cb44b",  # green
    "#f58231",  # orange
    "#911eb4",  # purple
    "#46f0f0",  # cyan
    "#f032e6",  # magenta
    "#bcf60c",  # lime
    "#9a6324",  # brown
    "#008080",  # teal
    "#fabebe",  # pink
    "#808000",  # olive
]


def _next_palette_color(existing_colors: list[str]) -> str:
    used = {c.lower() for c in existing_colors}
    for c in CLASS_PALETTE:
        if c.lower() not in used:
            return c
    return CLASS_PALETTE[len(existing_colors) % len(CLASS_PALETTE)]

# (x_channel_idx, y_channel_idx) into BGR mean array, plus titles/labels
PROJECTIONS = [(2, 1), (1, 0), (0, 2)]
TITLES = ["R vs G", "G vs B", "B vs R"]
AXIS_LABELS = [
    ("R contrast", "G contrast"),
    ("G contrast", "B contrast"),
    ("B contrast", "R contrast"),
]


def _hex_to_rgb_floats(hex_str: str) -> tuple[float, float, float]:
    s = hex_str.lstrip("#")
    if len(s) != 6:
        return (0.5, 0.5, 0.5)
    return (
        int(s[0:2], 16) / 255.0,
        int(s[2:4], 16) / 255.0,
        int(s[4:6], 16) / 255.0,
    )


def _hex_to_bgr_tuple(hex_str: str) -> tuple[int, int, int]:
    s = hex_str.lstrip("#")
    if len(s) != 6:
        return (0, 255, 0)
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)


def _bgr_to_qpixmap(image_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class FlakePreviewDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Tool)
        self.setWindowTitle("Flake preview")
        self.setMinimumSize(280, 280)
        self._image_label = QLabel(alignment=Qt.AlignCenter)
        self._image_label.setStyleSheet("background: #222;")
        self._caption_label = QLabel("")
        self._caption_label.setWordWrap(True)
        self._caption_label.setStyleSheet("color: #555; padding: 4px;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self._image_label, 1)
        lay.addWidget(self._caption_label)

    def show_flake(
        self,
        image_bgr: np.ndarray,
        instance_mask: np.ndarray,
        label: int,
        caption: str,
        contour_color_hex: str,
    ) -> None:
        component = (instance_mask == label).astype(np.uint8)
        ys, xs = np.where(component)
        if len(xs) == 0:
            self._image_label.setText("Could not find this flake in its mask.")
            self._caption_label.setText(caption)
            return
        h_img, w_img = component.shape
        x0 = max(0, int(xs.min()) - PREVIEW_PADDING_PX)
        y0 = max(0, int(ys.min()) - PREVIEW_PADDING_PX)
        x1 = min(w_img, int(xs.max()) + 1 + PREVIEW_PADDING_PX)
        y1 = min(h_img, int(ys.max()) + 1 + PREVIEW_PADDING_PX)
        crop = image_bgr[y0:y1, x0:x1].copy()
        comp_crop = component[y0:y1, x0:x1]
        contours, _ = cv2.findContours(
            comp_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(crop, contours, -1, _hex_to_bgr_tuple(contour_color_hex), 2)

        pix = _bgr_to_qpixmap(crop)
        if max(pix.width(), pix.height()) > PREVIEW_MAX_SIDE:
            pix = pix.scaled(
                PREVIEW_MAX_SIDE,
                PREVIEW_MAX_SIDE,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        self._image_label.setPixmap(pix)
        self._caption_label.setText(caption)
        self.adjustSize()


class SemanticView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.project: MaterialProject | None = None

        self._records: list[InstanceContrast] = []
        self._cluster_ids: np.ndarray | None = None
        self._means: np.ndarray | None = None
        self._dirty: bool = False
        self._active_class_id: int = 0  # 0 = Unassigned

        self._fig = None
        self._axes: list = []
        self._scatters: list = []
        self._lassos: list = []
        self._canvas = None
        self._pick_cid: int | None = None
        self._preview_dialog: FlakePreviewDialog | None = None
        self._image_cache: "OrderedDict[str, tuple[np.ndarray, np.ndarray]]" = (
            OrderedDict()
        )
        self._flatfield_img: np.ndarray | None = None
        self._flatfield_loaded_for: str | None = None

        self.setFocusPolicy(Qt.StrongFocus)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Main column: toolbar + plots + status
        main = QWidget()
        ml = QVBoxLayout(main)
        ml.setContentsMargins(8, 8, 8, 8)

        toolbar = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        self.save_btn = QPushButton("Save (S)")
        self.save_btn.clicked.connect(self.save)
        toolbar.addWidget(self.refresh_btn)
        toolbar.addWidget(self.save_btn)
        toolbar.addStretch(1)
        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("color: #555;")
        toolbar.addWidget(self.summary_label)
        ml.addLayout(toolbar)

        self._plot_container = QWidget()
        self._plot_layout = QVBoxLayout(self._plot_container)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(self._plot_container, 1)

        self.empty_label = QLabel(
            "Annotate at least one image first (Instance Masks tab), "
            "then press Refresh."
        )
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("color: #888; font-size: 14px;")
        ml.addWidget(self.empty_label)

        self.status_label = QLabel(self._default_status())
        self.status_label.setStyleSheet("color: #555; padding: 6px;")
        self.status_label.setWordWrap(True)
        ml.addWidget(self.status_label)
        outer.addWidget(main, 1)

        # Right: class palette
        right = QWidget()
        right.setFixedWidth(220)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 8, 8, 8)
        rl.addWidget(QLabel("Classes"))
        self.class_list = QListWidget()
        self.class_list.itemSelectionChanged.connect(self._on_class_changed)
        rl.addWidget(self.class_list, 1)
        cls_btns = QHBoxLayout()
        self.add_class_btn = QPushButton("+ Add")
        self.add_class_btn.clicked.connect(self._add_class)
        self.del_class_btn = QPushButton("− Remove")
        self.del_class_btn.clicked.connect(self._remove_class)
        cls_btns.addWidget(self.add_class_btn)
        cls_btns.addWidget(self.del_class_btn)
        rl.addLayout(cls_btns)
        outer.addWidget(right)

        for key, slot in (
            (Qt.Key_A, self._prev_class),
            (Qt.Key_D, self._next_class),
            (Qt.Key_X, lambda: self._select_class_index(0)),
            (Qt.Key_S, self.save),
        ):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)
        for n in range(0, 10):
            sc = QShortcut(QKeySequence(str(n)), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda idx=n: self._select_class_index(idx))

    # ---------- public ----------

    def set_project(self, project: MaterialProject) -> None:
        self.project = project
        self._records = []
        self._cluster_ids = None
        self._means = None
        self._dirty = False
        self._refresh_palette()
        self.summary_label.setText("")
        if self._canvas is not None:
            for ax in self._axes:
                ax.clear()
            self._canvas.draw_idle()
            self._canvas.hide()
        self.empty_label.show()

    def refresh(self) -> None:
        if self.project is None:
            return
        if self._dirty:
            self.save()
        self.summary_label.setText("Computing contrasts…")
        self.refresh_btn.setEnabled(False)
        self.repaint()
        try:
            self._records = aggregate_project_contrasts(self.project)
        finally:
            self.refresh_btn.setEnabled(True)
        self._means = (
            np.array([r.mean_bgr for r in self._records], dtype=np.float64)
            if self._records
            else None
        )
        self._cluster_ids = np.zeros(len(self._records), dtype=np.int32)
        self._load_existing_assignments()
        self._dirty = False
        self._update_summary()
        self._render()

    def save(self) -> bool:
        if self.project is None or self._cluster_ids is None or not self._records:
            return False
        by_image: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for rec, cid in zip(self._records, self._cluster_ids):
            by_image[rec.image_name].append((rec.label, int(cid)))
        n_files = 0
        for image_name, items in by_image.items():
            image_path = self.project.path / image_name
            inst = load_instance_mask(
                mask_path_for_image(self.project.path, image_path)
            )
            if inst is None:
                continue
            sem = np.zeros(inst.shape, dtype=np.uint8)
            for label, cid in items:
                if cid > 0:
                    sem[inst == label] = cid
            sem_path = semantic_mask_path_for_image(
                self.project.path, image_path
            )
            if sem.max() == 0:
                if sem_path.exists():
                    sem_path.unlink()
            else:
                save_semantic_mask(sem_path, sem)
            n_files += 1
        self._dirty = False
        self._update_summary()
        self._set_status(f"Saved semantic masks for {n_files} image(s).")
        return True

    # ---------- internals: data ----------

    def _load_existing_assignments(self) -> None:
        if self.project is None or self._cluster_ids is None:
            return
        by_image: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, rec in enumerate(self._records):
            by_image[rec.image_name].append((i, rec.label))
        for image_name, items in by_image.items():
            image_path = self.project.path / image_name
            sem = load_semantic_mask(
                semantic_mask_path_for_image(self.project.path, image_path)
            )
            if sem is None:
                continue
            inst = load_instance_mask(
                mask_path_for_image(self.project.path, image_path)
            )
            if inst is None or inst.shape != sem.shape:
                continue
            for idx, label in items:
                pixels = sem[inst == label]
                if len(pixels) == 0:
                    continue
                vals, counts = np.unique(pixels, return_counts=True)
                self._cluster_ids[idx] = int(vals[np.argmax(counts)])

    # ---------- internals: palette ----------

    def _refresh_palette(self) -> None:
        self.class_list.blockSignals(True)
        self.class_list.clear()
        unassigned = QListWidgetItem(UNASSIGNED_LABEL)
        unassigned.setData(Qt.UserRole, 0)
        pix = QPixmap(16, 16)
        pix.fill(QColor(UNASSIGNED_COLOR))
        unassigned.setIcon(QIcon(pix))
        self.class_list.addItem(unassigned)
        if self.project is not None:
            for cls in self.project.classes:
                item = QListWidgetItem(cls.name)
                item.setData(Qt.UserRole, cls.id)
                p = QPixmap(16, 16)
                p.fill(QColor(cls.color))
                item.setIcon(QIcon(p))
                self.class_list.addItem(item)
        self.class_list.blockSignals(False)
        if self.class_list.currentRow() < 0:
            self.class_list.setCurrentRow(0)

    def _on_class_changed(self) -> None:
        item = self.class_list.currentItem()
        self._active_class_id = int(item.data(Qt.UserRole)) if item else 0

    def _prev_class(self) -> None:
        row = self.class_list.currentRow()
        if row > 0:
            self.class_list.setCurrentRow(row - 1)

    def _next_class(self) -> None:
        row = self.class_list.currentRow()
        if 0 <= row < self.class_list.count() - 1:
            self.class_list.setCurrentRow(row + 1)

    def _select_class_index(self, idx: int) -> None:
        if 0 <= idx < self.class_list.count():
            self.class_list.setCurrentRow(idx)

    def _add_class(self) -> None:
        if self.project is None:
            return
        name, ok = QInputDialog.getText(
            self, "New class", "Class name (e.g. 1L, 2L, Bulk):"
        )
        if not ok or not name.strip():
            return
        color = _next_palette_color([c.color for c in self.project.classes])
        new_cls = self.project.add_class(name.strip(), color)
        self._refresh_palette()
        for i in range(self.class_list.count()):
            if int(self.class_list.item(i).data(Qt.UserRole)) == new_cls.id:
                self.class_list.setCurrentRow(i)
                break
        self._update_colors()

    def _remove_class(self) -> None:
        item = self.class_list.currentItem()
        if item is None or self.project is None:
            return
        cid = int(item.data(Qt.UserRole))
        if cid == 0:
            return
        ans = QMessageBox.question(
            self,
            "Remove class?",
            f"Remove class '{item.text()}'? Instances currently assigned will "
            "fall back to 'Unassigned' in the plot. Pixels in already-saved masks "
            "keep that ID until you re-save.",
        )
        if ans != QMessageBox.Yes:
            return
        self.project.remove_class(cid)
        if self._cluster_ids is not None:
            self._cluster_ids[self._cluster_ids == cid] = 0
            self._dirty = True
        self._refresh_palette()
        self._update_colors()

    # ---------- internals: plots ----------

    def _ensure_canvas(self) -> None:
        if self._canvas is not None:
            return
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._fig = Figure(figsize=(15, 5), tight_layout=True)
        self._axes = [
            self._fig.add_subplot(131),
            self._fig.add_subplot(132),
            self._fig.add_subplot(133),
        ]
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._plot_layout.addWidget(self._canvas)
        self._pick_cid = self._canvas.mpl_connect(
            "pick_event", self._on_pick
        )

    def _render(self) -> None:
        if not self._records or self._means is None:
            if self._canvas is not None:
                self._canvas.hide()
            self.empty_label.show()
            return
        self.empty_label.hide()
        self._ensure_canvas()
        self._canvas.show()

        from matplotlib.widgets import LassoSelector

        for sel in self._lassos:
            try:
                sel.disconnect_events()
            except Exception:
                pass
        self._lassos = []
        self._scatters = []

        colors = self._compute_colors()
        for i, ax in enumerate(self._axes):
            ax.clear()
            x_idx, y_idx = PROJECTIONS[i]
            sc = ax.scatter(
                self._means[:, x_idx],
                self._means[:, y_idx],
                c=colors,
                s=40,
                edgecolor="none",
                picker=True,
                pickradius=6,
            )
            self._scatters.append(sc)
            ax.set_xlabel(AXIS_LABELS[i][0])
            ax.set_ylabel(AXIS_LABELS[i][1])
            ax.set_title(TITLES[i])
            ax.axhline(0, color="#444", lw=0.8, ls="--", zorder=0)
            ax.axvline(0, color="#444", lw=0.8, ls="--", zorder=0)
            ax.grid(True, ls=":", lw=0.5, alpha=0.5)
            sel = LassoSelector(
                ax, lambda verts, ax_idx=i: self._on_lasso(verts, ax_idx)
            )
            self._lassos.append(sel)

        self._canvas.draw_idle()

    def _on_lasso(self, verts, ax_idx: int) -> None:
        if self._cluster_ids is None or self._means is None:
            return
        from matplotlib.path import Path as MplPath

        x_idx, y_idx = PROJECTIONS[ax_idx]
        coords = self._means[:, [x_idx, y_idx]]
        path = MplPath(verts)
        inside = path.contains_points(coords)
        if not inside.any():
            return
        self._cluster_ids[inside] = self._active_class_id
        self._dirty = True
        self._update_colors()
        name = self._active_class_name()
        self._set_status(
            f"Assigned {int(inside.sum())} instance(s) to '{name}'. "
            "Press S to save."
        )

    def _update_colors(self) -> None:
        if not self._scatters or self._cluster_ids is None:
            return
        colors = self._compute_colors()
        for sc in self._scatters:
            sc.set_color(colors)
        if self._canvas is not None:
            self._canvas.draw_idle()
        self._update_summary()

    def _compute_colors(self) -> list:
        out = []
        if self._cluster_ids is None:
            return out
        for cid in self._cluster_ids:
            if cid == 0:
                out.append((0.5, 0.5, 0.5, 0.5))
            else:
                cls = (
                    self.project.find_class(int(cid))
                    if self.project is not None
                    else None
                )
                if cls is None:
                    out.append((0.3, 0.3, 0.3, 0.4))
                else:
                    rgb = _hex_to_rgb_floats(cls.color)
                    out.append((*rgb, 0.9))
        return out

    def _active_class_name(self) -> str:
        if self._active_class_id == 0 or self.project is None:
            return UNASSIGNED_LABEL
        cls = self.project.find_class(self._active_class_id)
        return cls.name if cls else UNASSIGNED_LABEL

    def _update_summary(self) -> None:
        n_total = len(self._records)
        n_images = len({r.image_name for r in self._records})
        if self._cluster_ids is None or n_total == 0:
            self.summary_label.setText("")
            return
        n_assigned = int((self._cluster_ids > 0).sum())
        text = (
            f"{n_assigned}/{n_total} assigned · {n_images} image"
            f"{'s' if n_images != 1 else ''}"
        )
        if self._dirty:
            text += " · unsaved"
        self.summary_label.setText(text)

    def _set_status(self, msg: str) -> None:
        self.status_label.setText(msg)

    @staticmethod
    def _default_status() -> str:
        return (
            "Lasso a cluster on any plot to assign the active class. "
            "Click a single dot to preview that flake. "
            "A/D switch class · X = Unassigned · 0-9 jump to class · S save · Refresh recomputes."
        )

    # ---------- internals: flake preview ----------

    def _get_flatfield(self) -> np.ndarray | None:
        if self.project is None:
            return None
        if not self.project.flatfield_enabled:
            return None
        ff_path = self.project.flatfield_full_path
        if ff_path is None or not ff_path.exists():
            return None
        key = str(ff_path)
        if self._flatfield_loaded_for == key and self._flatfield_img is not None:
            return self._flatfield_img
        try:
            self._flatfield_img = load_flatfield(ff_path)
            self._flatfield_loaded_for = key
        except Exception:
            self._flatfield_img = None
            self._flatfield_loaded_for = None
        return self._flatfield_img

    def _load_image_and_mask(
        self, image_name: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if self.project is None:
            return None
        cached = self._image_cache.get(image_name)
        if cached is not None:
            self._image_cache.move_to_end(image_name)
            return cached

        image_path = self.project.path / image_name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return None
        flatfield = self._get_flatfield()
        if flatfield is not None and flatfield.shape == image.shape:
            try:
                image = remove_vignette(image, flatfield)
            except Exception:
                pass
        inst_path = mask_path_for_image(self.project.path, image_path)
        instance_mask = load_instance_mask(inst_path)
        if instance_mask is None or instance_mask.shape != image.shape[:2]:
            return None

        self._image_cache[image_name] = (image, instance_mask)
        while len(self._image_cache) > IMAGE_CACHE_SIZE:
            self._image_cache.popitem(last=False)
        return image, instance_mask

    def _on_pick(self, event) -> None:
        ind = getattr(event, "ind", None)
        if ind is None or len(ind) == 0:
            return
        idx = int(ind[0])
        if not (0 <= idx < len(self._records)):
            return
        rec = self._records[idx]
        loaded = self._load_image_and_mask(rec.image_name)
        if loaded is None:
            self._set_status(
                f"Could not load image or mask for '{rec.image_name}'."
            )
            return
        image_bgr, instance_mask = loaded

        cid = (
            int(self._cluster_ids[idx])
            if self._cluster_ids is not None
            else 0
        )
        if cid == 0:
            class_label = UNASSIGNED_LABEL
            color_hex = "#ffff00"  # yellow — high contrast for unassigned
        else:
            cls = (
                self.project.find_class(cid) if self.project is not None else None
            )
            if cls is None:
                class_label = f"Class {cid}"
                color_hex = "#ffff00"
            else:
                class_label = cls.name
                color_hex = cls.color

        b, g, r = rec.mean_bgr
        caption = (
            f"{rec.image_name} · label {rec.label}\n"
            f"Current class: {class_label} · {rec.pixel_count:,} px\n"
            f"Mean contrast — R: {r:+.3f}  G: {g:+.3f}  B: {b:+.3f}"
        )

        if self._preview_dialog is None:
            self._preview_dialog = FlakePreviewDialog(self)
        self._preview_dialog.show_flake(
            image_bgr, instance_mask, rec.label, caption, color_hex
        )
        if not self._preview_dialog.isVisible():
            self._preview_dialog.move(QCursor.pos() + QPoint(20, 20))
        self._preview_dialog.show()
        self._preview_dialog.raise_()
