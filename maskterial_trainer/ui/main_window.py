from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import UserConfig
from ..pipeline.project import MaterialProject
from .annotation_view import AnnotationView
from .convert_view import ConvertView
from .evaluation_view import EvaluationView
from .export_view import ExportView
from .images_view import ImagesView
from .semantic_view import SemanticView
from .training_view import TrainingView
from .welcome_view import WelcomeView

STAGES = [
    "Images",
    "Instance Masks",
    "Semantic Masks",
    "COCO Conversion",
    "Train",
    "Evaluate",
    "Export",
]
ENABLED_NOW = {
    "Images",
    "Instance Masks",
    "Semantic Masks",
    "COCO Conversion",
    "Train",
    "Evaluate",
    "Export",
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MaskTerial Trainer")
        self.resize(1280, 800)
        self.config = UserConfig.load()
        self.project: MaterialProject | None = None

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sidebar = QWidget()
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet("background-color: #f4f4f4;")
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(12, 16, 12, 16)
        self.project_label = QLabel("No project")
        self.project_label.setStyleSheet("font-weight: bold;")
        self.project_label.setWordWrap(True)
        sb_layout.addWidget(self.project_label)
        self.switch_btn = QPushButton("Switch project…")
        self.switch_btn.clicked.connect(self._show_welcome)
        sb_layout.addWidget(self.switch_btn)
        sb_layout.addSpacing(16)

        self.stage_list = QListWidget()
        self.stage_list.setStyleSheet("QListWidget { background: transparent; }")
        self.stage_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.stage_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.stage_list.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.stage_list.setUniformItemSizes(True)
        for stage in STAGES:
            self.stage_list.addItem(QListWidgetItem(stage))
        self.stage_list.currentRowChanged.connect(self._on_stage_change)
        sb_layout.addWidget(self.stage_list)
        sb_layout.addStretch(1)
        layout.addWidget(sidebar)

        self.stack = QStackedWidget()
        self.welcome = WelcomeView(self.config)
        self.welcome.folder_selected.connect(self._open_or_create_project)
        self.images_view = ImagesView()
        self.annotation_view = AnnotationView()
        self.semantic_view = SemanticView()
        self.convert_view = ConvertView()
        self.training_view = TrainingView()
        self.evaluation_view = EvaluationView()
        self.export_view = ExportView()

        self.stack.addWidget(self.welcome)
        self.stack.addWidget(self.images_view)
        self.stack.addWidget(self.annotation_view)
        self.stack.addWidget(self.semantic_view)
        self.stack.addWidget(self.convert_view)
        self.stack.addWidget(self.training_view)
        self.stack.addWidget(self.evaluation_view)
        self.stack.addWidget(self.export_view)
        layout.addWidget(self.stack, 1)

        self._show_welcome()

    @staticmethod
    def _placeholder(text: str) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #888; font-size: 16px;")
        lay.addWidget(label)
        return w

    def _show_welcome(self) -> None:
        self.stack.setCurrentWidget(self.welcome)
        self.stage_list.clearSelection()
        self.welcome.refresh()

    def _set_project(self, project: MaterialProject) -> None:
        self.project = project
        self.project_label.setText(project.name)
        self.images_view.set_project(project)
        self.annotation_view.set_project(project)
        self.semantic_view.set_project(project)
        self.convert_view.set_project(project)
        self.training_view.set_project(project)
        self.evaluation_view.set_project(project)
        self.export_view.set_project(project)
        self.config.add_recent(project.path)
        for i in range(self.stage_list.count()):
            item = self.stage_list.item(i)
            flags = item.flags()
            if item.text() in ENABLED_NOW:
                item.setFlags(flags | Qt.ItemIsEnabled)
            else:
                item.setFlags(flags & ~Qt.ItemIsEnabled)
        self.stage_list.setCurrentRow(0)

    def _on_stage_change(self, row: int) -> None:
        if row < 0 or self.project is None:
            return
        self.stack.setCurrentIndex(row + 1)
        stage = STAGES[row]
        if stage == "Instance Masks":
            self.annotation_view.refresh()
            self.annotation_view.canvas.setFocus()
        elif stage == "Semantic Masks":
            self.semantic_view.refresh()
            self.semantic_view.setFocus()
        elif stage == "Train":
            self.training_view.refresh()
        elif stage == "Evaluate":
            self.evaluation_view.refresh()
        elif stage == "Export":
            self.export_view.refresh()

    def _open_or_create_project(self, path: str) -> None:
        try:
            project = MaterialProject.open_or_create(Path(path))
        except Exception as e:
            QMessageBox.critical(self, "Could not open project", str(e))
            return
        self._set_project(project)
