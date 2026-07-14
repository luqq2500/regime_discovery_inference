"""
Interactive traversal curves: asset % change vs sigma sweep.
"""
from __future__ import annotations

from typing import Optional, List
import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QCheckBox, QFrame, QSizePolicy
)
import pyqtgraph as pg

from view.viewmodels.analysis_vm import AnalysisViewModel


# Vibrant blue + greys for multiple series
COLORS = [
    "#00b4ff", "#ff6b6b", "#51cf66", "#fcc419", "#cc5de8",
    "#20c997", "#ff922b", "#339af0", "#e599f7", "#94d82d",
    "#ff8787", "#69db7c", "#ffe066", "#b197fc", "#63e6be",
]


class TraversalPlotWidget(QWidget):
    def __init__(self, vm: AnalysisViewModel, parent=None):
        super().__init__(parent)
        self.vm = vm
        self._curves = []
        self._build_ui()
        self._connect()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Controls
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Regime:"))
        self.cmb_regime = QComboBox()
        self.cmb_regime.setMinimumWidth(120)
        ctrl.addWidget(self.cmb_regime)

        ctrl.addWidget(QLabel("  Focus Asset:"))
        self.cmb_asset = QComboBox()
        self.cmb_asset.setMinimumWidth(160)
        self.cmb_asset.addItem("All (Top filtered)")
        ctrl.addWidget(self.cmb_asset)

        self.chk_legend = QCheckBox("Legend")
        self.chk_legend.setChecked(True)
        ctrl.addWidget(self.chk_legend)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Plot
        pg.setConfigOptions(antialias=True, background="#0f0f0f", foreground="#e0e0e0")
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#0f0f0f")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "σ (latent units)")
        self.plot.setLabel("left", "% Change (raw)" if self.vm.state.use_raw else "Scaled")
        self.plot.addLegend(offset=(10, 10))
        self.plot.setMinimumHeight(380)
        layout.addWidget(self.plot)

        # Info bar
        self.info = QLabel("Run analysis to see traversal curves.")
        self.info.setObjectName("subtitle")
        layout.addWidget(self.info)

    def _connect(self):
        self.vm.analysis_finished.connect(self._on_finished)
        self.vm.state_changed.connect(self._refresh_if_needed)
        self.cmb_regime.currentIndexChanged.connect(self._redraw)
        self.cmb_asset.currentIndexChanged.connect(self._redraw)
        self.chk_legend.toggled.connect(self._toggle_legend)

    def _on_finished(self, response):
        self.cmb_regime.blockSignals(True)
        self.cmb_regime.clear()
        for out in response.outputs:
            dof = out.degree_of_freedom
            self.cmb_regime.addItem(f"Regime {out.regime}  (ν={dof:.1f})", out.regime)
        self.cmb_regime.blockSignals(False)

        # Populate assets from first output filtered
        self.cmb_asset.blockSignals(True)
        self.cmb_asset.clear()
        self.cmb_asset.addItem("All (Top filtered)", None)
        if response.outputs:
            cols = [c for c in response.outputs[0].result_raw.columns if c != "sweep"]
            for c in cols:
                self.cmb_asset.addItem(c, c)
        self.cmb_asset.blockSignals(False)

        self._redraw()

    def _refresh_if_needed(self):
        if self.vm.state.response is not None:
            self._redraw()

    def _toggle_legend(self, on: bool):
        legend = self.plot.plotItem.legend
        if legend:
            legend.setVisible(on)

    def _redraw(self):
        self.plot.clear()
        # re-add legend
        if self.plot.plotItem.legend is None:
            self.plot.addLegend(offset=(10, 10))
        else:
            self.plot.plotItem.legend.clear()

        regime = self.cmb_regime.currentData()
        if regime is None and self.cmb_regime.count() > 0:
            regime = self.cmb_regime.itemData(0)
        if regime is None:
            return

        df = self.vm.get_traversal_df(regime)
        if df is None or df.empty:
            self.info.setText("No data for selected regime.")
            return

        filtered = self.vm.get_filtered_stats(regime)
        focus = self.cmb_asset.currentData()

        sweeps = df["sweep"].values if "sweep" in df.columns else np.arange(len(df))

        # Determine which columns to plot
        if focus:
            cols = [focus] if focus in df.columns else []
        else:
            if filtered is not None and not filtered.empty:
                # filtered index are the asset names
                cols = [c for c in filtered.index if c in df.columns][:12]
            else:
                cols = [c for c in df.columns if c != "sweep"][:8]

        for i, col in enumerate(cols):
            color = COLORS[i % len(COLORS)]
            pen = pg.mkPen(color=color, width=2)
            y = df[col].values
            self.plot.plot(sweeps, y, pen=pen, name=col, symbol=None)

        unit = "% Change" if self.vm.state.use_raw else "Scaled latent"
        self.plot.setLabel("left", unit)
        self.plot.setTitle(f"Regime {regime} — Latent Traversal", color="#00b4ff", size="12pt")

        out = self.vm.get_output_for_regime(regime)
        dof = out.degree_of_freedom if out else 0
        self.info.setText(
            f"Regime {regime} | ν={dof:.2f} | σ ∈ [{self.vm.state.sigma_min}, {self.vm.state.sigma_max}] | "
            f"Showing {len(cols)} series | {'Raw %' if self.vm.state.use_raw else 'Scaled'}"
        )
