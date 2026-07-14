"""
Top assets diagnostics table with mean / std / SNR.
Supports automatic numeric and alphabetical column sorting.
"""
from __future__ import annotations


from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QPushButton, QFileDialog, QMessageBox
)

from view.viewmodels.analysis_vm import AnalysisViewModel


class NumericTableWidgetItem(QTableWidgetItem):
    """Subclass of QTableWidgetItem offering clean numeric comparisons for sorting."""

    def __init__(self, value: float, text: str):
        super().__init__(text)
        self.value = value

    def __lt__(self, other: QTableWidgetItem) -> bool:
        if isinstance(other, NumericTableWidgetItem):
            return self.value < other.value
        # Attempt raw numeric parsing as fallback
        try:
            return float(self.value) < float(other.text())
        except ValueError:
            return self.text() < other.text()


class DiagnosticsTableWidget(QWidget):
    def __init__(self, vm: AnalysisViewModel, parent=None):
        super().__init__(parent)
        self.vm = vm
        self._build_ui()
        self._connect()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Regime:"))
        self.cmb_regime = QComboBox()
        self.cmb_regime.setMinimumWidth(140)
        ctrl.addWidget(self.cmb_regime)

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItem("Filtered (Top SNR)", "filtered")
        self.cmb_mode.addItem("Full Stats", "full")
        ctrl.addWidget(self.cmb_mode)

        ctrl.addStretch()

        self.btn_export = QPushButton("Export CSV")
        ctrl.addWidget(self.btn_export)
        layout.addLayout(ctrl)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        # Formatting headers
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setStretchLastSection(False)
        header.setSectionsClickable(True)
        self.table.verticalHeader().setVisible(False)

        layout.addWidget(self.table)

        self.info = QLabel("Run analysis to populate diagnostics.")
        self.info.setObjectName("subtitle")
        layout.addWidget(self.info)

    def _connect(self):
        self.vm.analysis_finished.connect(self._on_finished)
        self.vm.state_changed.connect(self._maybe_refresh)
        self.cmb_regime.currentIndexChanged.connect(self._redraw)
        self.cmb_mode.currentIndexChanged.connect(self._redraw)
        self.btn_export.clicked.connect(self._export)

    def _on_finished(self, response):
        self.cmb_regime.blockSignals(True)
        self.cmb_regime.clear()
        for out in response.outputs:
            self.cmb_regime.addItem(f"Regime {out.regime}", out.regime)
        self.cmb_regime.blockSignals(False)
        self._redraw()

    def _maybe_refresh(self):
        if self.vm.state.response:
            self._redraw()

    def _redraw(self):
        # Disable sorting temporarily to block updates during population
        self.table.setSortingEnabled(False)

        regime = self.cmb_regime.currentData()
        if regime is None:
            self.table.setRowCount(0)
            return

        mode = self.cmb_mode.currentData()
        if mode == "filtered":
            df = self.vm.get_filtered_stats(regime)
        else:
            df = self.vm.get_full_stats(regime)

        if df is None or df.empty:
            self.table.setRowCount(0)
            self.info.setText("No statistics available.")
            return

        display_df = df.copy()
        if display_df.index.name is None or display_df.index.name != "asset":
            display_df = display_df.reset_index()
            if "index" in display_df.columns:
                display_df = display_df.rename(columns={"index": "asset"})

        cols = list(display_df.columns)
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels([str(c) for c in cols])
        self.table.setRowCount(len(display_df))

        for r, (_, row) in enumerate(display_df.iterrows()):
            for c, col in enumerate(cols):
                val = row[col]
                if isinstance(val, (float, int)) and not isinstance(val, bool):
                    text = f"{val:.4f}" if abs(val) < 1000 else f"{val:.2f}"
                    item = NumericTableWidgetItem(float(val), text)
                else:
                    text = str(val)
                    item = QTableWidgetItem(text)

                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                # Highlight high SNR items
                if col.lower() == "snr" and isinstance(val, (int, float)) and val >= 0.5:
                    item.setForeground(Qt.GlobalColor.cyan)
                self.table.setItem(r, c, item)

        # Safe to re-enable sorting once rows are established
        self.table.setSortingEnabled(True)

        self.info.setText(
            f"Regime {regime} | {mode} | {'Raw' if self.vm.state.use_raw else 'Scaled'} | "
            f"{len(display_df)} assets (Click header columns to sort)"
        )

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "regime_stats.csv", "CSV (*.csv)")
        if not path:
            return
        regime = self.cmb_regime.currentData()
        mode = self.cmb_mode.currentData()
        df = self.vm.get_filtered_stats(regime) if mode == "filtered" else self.vm.get_full_stats(regime)
        if df is not None:
            df.to_csv(path)
            QMessageBox.information(self, "Export", f"Saved to {path}")