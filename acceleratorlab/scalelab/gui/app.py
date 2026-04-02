"""Entry point for the AcceleratorLab desktop GUI."""
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore    import Qt
from scalelab.gui.theme       import DARK
from scalelab.gui.main_window import MainWindow


def run():
    app = QApplication(sys.argv)
    app.setApplicationName("AcceleratorLab Console Pro")
    app.setOrganizationName("AcceleratorLab")
    app.setStyleSheet(DARK)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
