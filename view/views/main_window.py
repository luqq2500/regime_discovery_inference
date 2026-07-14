"""
Main application window.
Assembles ConfigPanel + Tabbed content (Traversal / Diagnostics / Comparison / Embeddings).
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTabWidget,
    QSplitter, QStatusBar, QLabel, QFrame
)

from view.viewmodels.analysis_vm import AnalysisViewModel
from view.viewmodels.repository_vm import RepositoryViewModel
from view.views.regime_view import RegimeOverview
from view.views.traversal_explorer import TraversalExplorer
from view.views.config_panel import ConfigPanel
from view.views.diagnostics_table import DiagnosticsTableWidget
from view.views.embeddings_view import EmbeddingsView


class MainWindow(QMainWindow):
    def __init__(
        self,
        analysis_vm: AnalysisViewModel,
        repo_vm: RepositoryViewModel,
        parent=None,
    ):
        super().__init__(parent)
        self.avm = analysis_vm
        self.rvm = repo_vm

        self.setWindowTitle("Regime Discovery — Latent Traversal Analysis")
        self.resize(1400, 860)
        self.setMinimumSize(1100, 700)

        self._build_ui()
        self._connect()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left: Config
        self.config = ConfigPanel(self.avm)
        main_layout.addWidget(self.config)

        # Right: Tabs
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        # Changed tab title from "Sunburst Overview" to "Regime Overview"
        self.tab_regime = RegimeOverview(self.avm)
        self.tabs.addTab(self.tab_regime, "Regime Overview")

        self.tab_traversal = TraversalExplorer(self.avm)
        self.tabs.addTab(self.tab_traversal, "Traversal Explorer")

        self.tab_diag = DiagnosticsTableWidget(self.avm)
        self.tabs.addTab(self.tab_diag, "Diagnostics")

        self.tab_embed = EmbeddingsView(self.rvm)
        self.tabs.addTab(self.tab_embed, "Repository Explorer")

        right_l.addWidget(self.tabs)
        main_layout.addWidget(right, stretch=1)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — select regimes and click Run Analysis")

    def _connect(self):
        self.avm.analysis_started.connect(
            lambda: self.status.showMessage("Running latent traversal...")
        )
        self.avm.analysis_finished.connect(
            lambda r: self.status.showMessage(
                f"Done — {len(r.outputs)} regime(s) analysed"
            )
        )
        self.avm.analysis_failed.connect(
            lambda e: self.status.showMessage(f"Failed: {e[:100]}")
        )
        self.avm.progress.connect(self.status.showMessage)