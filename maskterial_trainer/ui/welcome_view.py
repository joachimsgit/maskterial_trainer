from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..config import UserConfig


class WelcomeView(QWidget):
    folder_selected = Signal(str)

    def __init__(self, config: UserConfig) -> None:
        super().__init__()
        self.config = config

        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 48, 48, 48)

        title = QLabel("MaskTerial Trainer")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)
        subtitle = QLabel(
            "Select a folder containing your microscope images. "
            "Project files will be created there if they don't exist yet."
        )
        subtitle.setStyleSheet("color: #666;")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        layout.addSpacing(24)

        btn_row = QHBoxLayout()
        select_btn = QPushButton("Select project folder…")
        select_btn.clicked.connect(self._on_select)
        btn_row.addWidget(select_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        layout.addSpacing(24)

        layout.addWidget(QLabel("Recent projects"))
        self.recent_list = QListWidget()
        self.recent_list.itemDoubleClicked.connect(self._on_recent_activated)
        layout.addWidget(self.recent_list, 1)

    def refresh(self) -> None:
        self.recent_list.clear()
        for p in self.config.recent_projects:
            if Path(p).exists():
                self.recent_list.addItem(QListWidgetItem(p))

    def _on_select(self) -> None:
        start = (
            self.config.recent_projects[0]
            if self.config.recent_projects
            and Path(self.config.recent_projects[0]).exists()
            else self.config.projects_root
        )
        path = QFileDialog.getExistingDirectory(
            self, "Select folder with images", start
        )
        if path:
            self.folder_selected.emit(path)

    def _on_recent_activated(self, item: QListWidgetItem) -> None:
        self.folder_selected.emit(item.text())
