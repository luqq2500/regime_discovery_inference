"""
Regime Overview visualization for hierarchical regime impact.
First tab — high-level overview. Fits full parent height responsively.
"""

from __future__ import annotations

from typing import Optional
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton,
    QSizePolicy, QHBoxLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtWebEngineWidgets import QWebEngineView

import plotly.express as px
import plotly.io as pio

from view.viewmodels.analysis_vm import AnalysisViewModel


class RegimeOverview(QWidget):
    def __init__(self, vm: AnalysisViewModel, parent=None):
        super().__init__(parent)
        self.vm = vm
        self.web_view = QWebEngineView()
        self._build_ui()
        self._connect()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Controls row
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Metric:"))
        self.cmb_metric = QComboBox()
        self.cmb_metric.addItems(["Mean Impact (%)", "SNR", "Std Dev"])
        ctrl.addWidget(self.cmb_metric)

        self.btn_refresh = QPushButton("Refresh Charts")
        ctrl.addWidget(self.btn_refresh)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Ensure the web view stretches aggressively to occupy all vertical space
        self.web_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.web_view, stretch=1)

        self.info = QLabel("Run analysis then click Refresh. Hierarchical view: Regime → Asset → Horizon")
        self.info.setObjectName("subtitle")
        layout.addWidget(self.info)

    def _connect(self):
        self.btn_refresh.clicked.connect(self._draw_view)
        self.vm.analysis_finished.connect(self._draw_view)
        self.cmb_metric.currentIndexChanged.connect(self._draw_view)

    def _draw_view(self):
        if not self.vm.state.response:
            return

        metric = self.cmb_metric.currentText()
        # Build hierarchical data
        data = []
        for out in self.vm.get_all_outputs():
            regime = f"Regime {out.regime} (ν={out.degree_of_freedom:.1f})"
            stats = out.filtered_asset_stats_raw if self.vm.state.use_raw else out.filtered_asset_stats_scaled
            if stats is None or stats.empty:
                continue
            for asset_idx, row in stats.iterrows():
                asset = str(asset_idx)
                val = float(row.get('mean', 0.0))
                if metric == "SNR":
                    val = float(row.get('snr', 0.0))
                elif metric == "Std Dev":
                    val = float(row.get('std', 0.0))

                # Split asset-horizon if possible
                if '-' in asset:
                    parts = asset.split('-', 1)
                    asset_name = parts[0]
                    horizon = parts[1] if len(parts) > 1 else 'All'
                else:
                    asset_name = asset
                    horizon = 'All'

                data.append({
                    "regime": regime,
                    "asset": asset_name,
                    "horizon": horizon,
                    "value": abs(val) if metric != "Mean Impact (%)" else val,
                    "label": f"{asset_name} {horizon}"
                })

        if not data:
            self.info.setText("No data for overview.")
            return

        df = pd.DataFrame(data)

        if metric == "Mean Impact (%)":
            # Helper to generate custom RGBA values for clean dual-color matching
            def get_rgba(color_str: str, alpha: float) -> str:
                if color_str.startswith('rgb('):
                    parts = color_str.replace('rgb(', '').replace(')', '').split(',')
                    r, g, b = parts[0].strip(), parts[1].strip(), parts[2].strip()
                    return f"rgba({r}, {g}, {b}, {alpha})"
                hex_str = color_str.lstrip('#')
                if len(hex_str) == 3:
                    hex_str = ''.join([c * 2 for c in hex_str])
                try:
                    r = int(hex_str[0:2], 16)
                    g = int(hex_str[2:4], 16)
                    b = int(hex_str[4:6], 16)
                    return f"rgba({r}, {g}, {b}, {alpha})"
                except ValueError:
                    return color_str

            # Create distinct classes based on direction of impact
            df['color_group'] = df.apply(
                lambda row: f"{row['regime']} (Positive)" if row['value'] >= 0 else f"{row['regime']} (Negative)",
                axis=1
            )

            unique_regimes = sorted(df['regime'].unique())
            palette = px.colors.qualitative.T10  # Standard clean palette

            # Programmatically map distinct opacities for cohesive positive/negative groups
            color_map = {}
            for idx, rg in enumerate(unique_regimes):
                base_color = palette[idx % len(palette)]
                color_map[f"{rg} (Positive)"] = get_rgba(base_color, 0.90)
                color_map[f"{rg} (Negative)"] = get_rgba(base_color, 0.45)

            df = df.sort_values(by="value", ascending=True)

            fig = px.bar(
                df,
                x='value',
                y='label',
                color='color_group',
                color_discrete_map=color_map,
                orientation='h',
                title=f"Regime Overview — {metric} (Bidirectional representation, shaded by direction)",
                labels={"value": "Mean Impact (%)", "label": "Asset & Horizon", "color_group": "Regime & Direction"},
                hover_data=["regime", "asset", "horizon"]
            )

            # Centered reference line at 0 for comparison
            fig.add_vline(x=0.0, line_dash="dash", line_color="#ff4d4d", line_width=1.5)

            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                paper_bgcolor='#0f0f0f',
                plot_bgcolor='#141414',
                font=dict(color='#e0e0e0', size=11),
                margin=dict(t=50, l=10, r=10, b=10),
                autosize=True,
            )
        else:
            fig = px.sunburst(
                df,
                path=['regime', 'asset', 'horizon'],
                values='value',
                title=f"Regime Impact Sunburst — {metric}",
                color='value',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                paper_bgcolor='#0f0f0f',
                font=dict(color='#e0e0e0', size=12),
                margin=dict(t=50, l=10, r=10, b=10),
                autosize=True,
            )

        # Export snippet to wrap inside our custom full viewport HTML wrapper
        plotly_html = pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs='cdn',
            config={'responsive': True}
        )

        # Viewport constraints enforced on body elements
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <style>
                html, body {{
                    margin: 0;
                    padding: 0;
                    width: 100%;
                    height: 100%;
                    background-color: #0f0f0f;
                    overflow: hidden;
                }}
                /* Force all plotly structural containers to stretch perfectly to 100% height */
                .plotly-graph-div, .plot-container, .svg-container {{
                    width: 100vw !important;
                    height: 100vh !important;
                }}
            </style>
        </head>
        <body>
            {plotly_html}
        </body>
        </html>
        """

        self.web_view.setHtml(full_html)
        self.info.setText(f"Paths rendered: {len(df)} | Metric: {metric}")