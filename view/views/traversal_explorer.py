"""
Unified Traversal Explorer (merged Curves + Comparison).
Rich card-based multi-select for regimes and assets.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QScrollArea, QFrame, QCheckBox, QGroupBox, QSizePolicy
)
import pyqtgraph as pg

from view.viewmodels.analysis_vm import AnalysisViewModel


class AssetCard(QFrame):
    clicked = pyqtSignal(str, bool)  # asset, selected

    def __init__(self, asset: str, display_metric_name: str, display_metric_val: float, is_selected: bool = False,
                 parent=None):
        super().__init__(parent)
        self.asset = asset
        self._selected = is_selected
        self.setObjectName("card")
        self.setMinimumHeight(65)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        self.name_label = QLabel(asset)
        self.name_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #ffffff;")
        layout.addWidget(self.name_label)

        self.metric_label = QLabel(f"{display_metric_name}: {display_metric_val:.3f}")
        self.metric_label.setStyleSheet("font-size: 10px; color: #a0a0a0;")
        layout.addWidget(self.metric_label)

        self._update_style()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._selected = not self._selected
            self._update_style()
            self.clicked.emit(self.asset, self._selected)

    def set_selected(self, selected: bool):
        self._selected = selected
        self._update_style()

    def _update_style(self):
        if self._selected:
            self.setStyleSheet("""
                QFrame#card {
                    border: 2px solid #00b4ff; 
                    background-color: #12283a;
                    border-radius: 4px;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame#card {
                    border: 1px solid #333333; 
                    background-color: #1a1a1a;
                    border-radius: 4px;
                }
                QFrame#card:hover {
                    border: 1px solid #555555;
                    background-color: #222222;
                }
            """)


class TraversalExplorer(QWidget):
    def __init__(self, vm: AnalysisViewModel, parent=None):
        super().__init__(parent)
        self.vm = vm
        self.selected_regimes: set[int] = set()
        self.selected_assets: set[str] = set()
        self._build_ui()
        self._connect()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Controls (Top Row)
        ctrl = QHBoxLayout()

        # Regimes Checkbox Area
        regime_box = QGroupBox("Active Regimes")
        rlayout = QHBoxLayout(regime_box)
        self.regime_check_group = QWidget()
        self.regime_layout = QHBoxLayout(self.regime_check_group)
        self.regime_layout.setContentsMargins(0, 0, 0, 0)
        rlayout.addWidget(self.regime_check_group)
        ctrl.addWidget(regime_box)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Horizontal Split: Sidebar (Assets) on Left, Plots on Right
        main_hbox = QHBoxLayout()

        # Sidebar Panel
        sidebar = QVBoxLayout()
        sidebar.setContentsMargins(0, 0, 0, 0)

        lbl_select = QLabel("Asset Selection")
        lbl_select.setStyleSheet("font-weight: bold; color: #e0e0e0;")
        sidebar.addWidget(lbl_select)

        # Dynamic Metric Selector inside Sidebar
        metric_layout = QHBoxLayout()
        metric_layout.addWidget(QLabel("Sort By:"))
        self.cmb_metric = QComboBox()
        self.cmb_metric.addItems(["SNR", "Mean Impact (%)", "Std Dev"])
        metric_layout.addWidget(self.cmb_metric)
        sidebar.addLayout(metric_layout)

        # Quick Actions
        btn_layout = QHBoxLayout()
        self.btn_select_all_assets = QPushButton("All")
        self.btn_select_none_assets = QPushButton("None")
        btn_layout.addWidget(self.btn_select_all_assets)
        btn_layout.addWidget(self.btn_select_none_assets)
        sidebar.addLayout(btn_layout)

        # Scrollable container for static Asset Cards
        self.scroll_assets = QScrollArea()
        self.scroll_assets.setWidgetResizable(True)
        self.scroll_assets.setMinimumWidth(200)
        self.scroll_assets.setMaximumWidth(260)

        self.scroll_container = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_container)
        self.scroll_layout.setContentsMargins(4, 4, 4, 4)
        self.scroll_layout.setSpacing(6)

        self.scroll_assets.setWidget(self.scroll_container)
        sidebar.addWidget(self.scroll_assets)

        main_hbox.addLayout(sidebar, stretch=0)

        # Plot Area
        plot_vbox = QVBoxLayout()
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#0f0f0f")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "σ")
        self.plot.setLabel("left", "% Change")
        self.plot.addLegend()
        self.plot.setMinimumHeight(420)
        plot_vbox.addWidget(self.plot)

        self.info = QLabel("Select regimes and assets to begin visual inspection.")
        self.info.setObjectName("subtitle")
        plot_vbox.addWidget(self.info)

        main_hbox.addLayout(plot_vbox, stretch=1)
        layout.addLayout(main_hbox)

    def _connect(self):
        self.vm.analysis_finished.connect(self._on_analysis_done)
        self.cmb_metric.currentIndexChanged.connect(self._on_metric_changed)
        self.vm.state_changed.connect(self._on_state_changed)
        self.btn_select_all_assets.clicked.connect(self._select_all_assets)
        self.btn_select_none_assets.clicked.connect(self._select_none_assets)

    def _on_analysis_done(self, response):
        self.selected_regimes.clear()

        # Repopulate regimes
        for i in reversed(range(self.regime_layout.count())):
            item = self.regime_layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)

        for out in response.outputs:
            cb = QCheckBox(f"R{out.regime} (ν={out.degree_of_freedom:.1f})")
            cb.setChecked(True)
            cb.toggled.connect(lambda checked, r=out.regime: self._toggle_regime(r, checked))
            self.regime_layout.addWidget(cb)
            self.selected_regimes.add(out.regime)

        self._populate_asset_cards()
        self._redraw()

    def _toggle_regime(self, regime: int, checked: bool):
        if checked:
            self.selected_regimes.add(regime)
        else:
            self.selected_regimes.discard(regime)
        self._populate_asset_cards()
        self._redraw()

    def _on_metric_changed(self):
        self._populate_asset_cards()
        self._redraw()

    def _on_state_changed(self):
        if self.vm.state.response:
            self._populate_asset_cards()
            self._redraw()

    def _populate_asset_cards(self):
        # Clear sidebar scroll widgets
        for i in reversed(range(self.scroll_layout.count())):
            item = self.scroll_layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)

        if not self.selected_regimes:
            return

        metric = self.cmb_metric.currentText()
        asset_stats: dict[str, list[float]] = {}

        # Aggregate metric values across selected active regimes
        for regime in self.selected_regimes:
            stats_df = self.vm.get_filtered_stats(regime)
            if stats_df is None or stats_df.empty:
                stats_df = self.vm.get_full_stats(regime)
            if stats_df is None or stats_df.empty:
                continue

            for idx, row in stats_df.iterrows():
                asset = str(idx)
                if metric == "SNR":
                    val = float(row.get('snr', 0.0))
                elif metric == "Std Dev":
                    val = float(row.get('std', 0.0))
                else:  # "Mean Impact (%)"
                    val = float(row.get('mean', 0.0))

                if asset not in asset_stats:
                    asset_stats[asset] = []
                asset_stats[asset].append(val)

        if not asset_stats:
            return

        # Average stats across active dimensions for stable multi-regime viewing
        averaged_stats = []
        for asset, vals in asset_stats.items():
            avg_val = float(np.mean(vals))
            averaged_stats.append((asset, avg_val))

        # Sort dynamically by metric
        if metric == "Mean Impact (%)":
            # Sort by absolute impact descending so strong negative/positive impacts prioritize top positions
            averaged_stats.sort(key=lambda x: abs(x[1]), reverse=True)
        else:
            averaged_stats.sort(key=lambda x: x[1], reverse=True)

        # Auto-initialize top 3 assets to ensure a visual plot is drawn automatically
        if not self.selected_assets:
            self.selected_assets = set(asset for asset, _ in averaged_stats[:3])

        # Render static cards
        for asset, val in averaged_stats:
            is_sel = asset in self.selected_assets
            card = AssetCard(asset, metric, val, is_selected=is_sel)
            card.clicked.connect(self._toggle_asset)
            self.scroll_layout.addWidget(card)

        # Pad layout to top alignment
        self.scroll_layout.addStretch()

    def _toggle_asset(self, asset: str, selected: bool):
        if selected:
            self.selected_assets.add(asset)
        else:
            self.selected_assets.discard(asset)
        self._redraw()

    def _select_all_assets(self):
        all_assets = set()
        for i in range(self.scroll_layout.count()):
            widget = self.scroll_layout.itemAt(i).widget()
            if isinstance(widget, AssetCard):
                widget.set_selected(True)
                all_assets.add(widget.asset)
        self.selected_assets = all_assets
        self._redraw()

    def _select_none_assets(self):
        for i in range(self.scroll_layout.count()):
            widget = self.scroll_layout.itemAt(i).widget()
            if isinstance(widget, AssetCard):
                widget.set_selected(False)
        self.selected_assets.clear()
        self._redraw()

    def _redraw(self):
        self.plot.clear()
        if self.plot.plotItem.legend:
            self.plot.plotItem.legend.clear()

        if not self.selected_regimes or not self.selected_assets:
            self.info.setText("Select active regimes and assets on the left to display plots.")
            return

        colors = ["#00b4ff", "#ff6b6b", "#51cf66", "#fcc419", "#cc5de8", "#20c997", "#ff922b", "#ae3ec9"]

        curve_count = 0
        for regime in sorted(self.selected_regimes):
            df = self.vm.get_traversal_df(regime)
            if df is None or df.empty:
                continue

            sweeps = df["sweep"].values if "sweep" in df.columns else np.arange(len(df))

            # Limit curve overlay lines for legibility
            for asset in sorted(self.selected_assets):
                if asset not in df.columns:
                    continue
                y = df[asset].values
                color = colors[curve_count % len(colors)]
                pen = pg.mkPen(color=color, width=2.5)
                name = f"R{regime} - {asset}"
                self.plot.plot(sweeps, y, pen=pen, name=name)
                curve_count += 1

        unit = "% Change (raw)" if self.vm.state.use_raw else "Scaled Latent Value"
        self.plot.setLabel("left", unit)
        self.plot.setTitle("Active Traversal Curves", color="#00b4ff", size="13pt")
        self.info.setText(f"Plotting {len(self.selected_regimes)} regime(s) × {len(self.selected_assets)} asset(s)")