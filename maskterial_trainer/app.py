import os
import sys

os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from .ui.main_window import MainWindow


def _apply_light_palette(app: QApplication) -> None:
    """Force a light palette regardless of the host OS theme.

    The views use hardcoded greys (`#555`, `#888`, etc.) and a light
    `#f4f4f4` sidebar that were tuned against a white background; on a
    dark-mode host those wash out. Pinning the palette here keeps the UI
    looking identical on every machine until the views are refactored
    to read all of their colours from the active palette.
    """
    p = QPalette()

    # Backgrounds
    p.setColor(QPalette.Window, QColor("#fafafa"))
    p.setColor(QPalette.Base, QColor("#ffffff"))
    p.setColor(QPalette.AlternateBase, QColor("#f4f4f4"))
    p.setColor(QPalette.Button, QColor("#f0f0f0"))
    p.setColor(QPalette.ToolTipBase, QColor("#ffffff"))

    # Foregrounds
    p.setColor(QPalette.WindowText, QColor("#202020"))
    p.setColor(QPalette.Text, QColor("#202020"))
    p.setColor(QPalette.ButtonText, QColor("#202020"))
    p.setColor(QPalette.ToolTipText, QColor("#202020"))
    p.setColor(QPalette.PlaceholderText, QColor("#888888"))
    p.setColor(QPalette.BrightText, QColor("#ff3030"))

    # Selection / accent
    p.setColor(QPalette.Highlight, QColor("#3478f6"))
    p.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.Link, QColor("#1565c0"))
    p.setColor(QPalette.LinkVisited, QColor("#7b1fa2"))

    # Disabled state — keep things readable but visibly muted.
    for role, color in (
        (QPalette.WindowText, "#909090"),
        (QPalette.Text, "#909090"),
        (QPalette.ButtonText, "#909090"),
        (QPalette.Highlight, "#cccccc"),
        (QPalette.HighlightedText, "#606060"),
    ):
        p.setColor(QPalette.Disabled, role, QColor(color))

    app.setPalette(p)


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    _apply_light_palette(app)
    app.setApplicationName("MaskTerial Trainer")
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
