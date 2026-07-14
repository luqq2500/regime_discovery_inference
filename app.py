#!/usr/bin/env python3
"""
Entry point for Regime Discovery UI.

Architecture:
  - core/ remains untouched
  - view/ contains all UI code (MVVM)
  - DI wiring happens here (only place that imports services)
"""
from __future__ import annotations

import sys
import os

from core.main import wire_lta_usecase, wire_repository

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from view.styles import DARK_STYLESHEET
from view.viewmodels.analysis_vm import AnalysisViewModel
from view.viewmodels.repository_vm import RepositoryViewModel
from view.views.main_window import MainWindow

def main():
    # High-DPI
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Regime Discovery")
    app.setOrganizationName("RegimeDiscovery")
    app.setStyle("Fusion")

    # Dark theme
    app.setStyleSheet(DARK_STYLESHEET)

    # Default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Wire core
    try:
        usecase = wire_lta_usecase()
        repository = wire_repository()
    except Exception as e:
        print(f"Failed to initialise core services: {e}")
        print("Make sure encoder.pt / decoder.pt / scaler.pt / repository.pt exist under core/service and core/repository.")
        sys.exit(1)

    # ViewModels
    analysis_vm = AnalysisViewModel(usecase, repository)
    repo_vm = RepositoryViewModel(repository)

    # Main Window
    window = MainWindow(analysis_vm, repo_vm)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
