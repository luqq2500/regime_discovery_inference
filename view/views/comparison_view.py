"""
Side-by-side / overlay comparison of multiple regimes.
"""
from __future__ import annotations

from typing import List
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox
)
import pyqtgraph as pg

from view.viewmodels.analysis_vm import AnalysisViewModel


COLORS = ["#00b4ff", "#ff6b6b", "#51cf66", "#fcc419", "#cc5de8", "#20c997"]


class ComparisonView(QWidget):
    def __init__(self, vm: AnalysisViewModel, parent=None):
        super().__init__(parent)
        self.vm = vm
        self._build_ui()
        self._connect()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Asset:"))
        self.cmb_asset = QComboBox()
        self.cmb_asset.setMinimumWidth(180)
        ctrl.addWidget(self.cmb_asset)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#0f0f0f")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "σ")
        self.plot.setLabel("left", "% Change")
        self.plot.addLegend()
        self.plot.setMinimumHeight(400)
        layout.addWidget(self.plot)

        self.info = QLabel("Select an asset after running multi-regime analysis.")
        self.info.setObjectName("subtitle")
        layout.addWidget(self.info)

    def _connect(self):
        self.vm.analysis_finished.connect(self._on_finished)
        self.vm.state_changed.connect(self._refresh)
        self.cmb_asset.currentIndexChanged.connect(self._redraw)

    def _on_finished(self, response):
        # Collect union of columns
        cols = set()
        for out in response.outputs:
            df = out.result_raw if self.vm.state.use_raw else out.result_scaled
            cols.update(c for c in df.columns if c != "sweep")
        self.cmb_asset.blockSignals(True)
        self.cmb_asset.clear()
        for c in sorted(cols):
            self.cmb_asset.addItem(c, c)
        self.cmb_asset.blockSignals(False)
        self._redraw()

    def _refresh(self):
        if self.vm.state.response:
            self._redraw()

    def _redraw(self):
        self.plot.clear()
        if self.plot.plotItem.legend:
            self.plot.plotItem.legend.clear()

        asset = self.cmb_asset.currentData()
        if not asset or not self.vm.state.response:
            return

        outputs = self.vm.get_all_outputs()
        for i, out in enumerate(outputs):
            df = out.result_raw if self.vm.state.use_raw else out.result_scaled
            if asset not in df.columns:
                continue
            sweeps = df["sweep"].values if "sweep" in df.columns else np.arange(len(df))
            y = df[asset].values
            color = COLORS[i % len(COLORS)]
            pen = pg.mkPen(color=color, width=2.5)
            self.plot.plot(sweeps, y, pen=pen, name=f"R{out.regime} (ν={out.degree_of_freedom:.1f})")

        self.plot.setTitle(f"Cross-Regime Comparison — {asset}", color="#00b4ff", size="12pt")
        self.info.setText(f"Comparing {len(outputs)} regimes on {asset}")
