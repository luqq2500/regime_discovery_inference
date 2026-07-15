"""
Repository explorer: UMAP 3D projection of latent embeddings.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTextEdit, QSplitter, QFrame, QCheckBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go
import plotly.io as pio

from view.viewmodels.repository_vm import RepositoryViewModel

# Distinct, high-contrast hex palette to ensure each of the 9 regimes
# retains a consistent, identifiable color across all combinations.
REGIME_COLORS = [
    "#E6194B",  # 1. Pure Crimson Red (Highly saturated, distinct from magenta)
    "#3CB44B",  # 2. Rich Emerald Green (Solid mid-tone weight)
    "#FFE119",  # 3. Vivid Gold Yellow (High-impact, adjusted for contrast)
    "#4363D8",  # 4. Electric Royal Blue (Strong anchoring tone)
    "#F58231",  # 5. Intense Safety Orange (Maximum visual pop)
    "#911DFF",  # 6. Deep Amethyst Purple (Pushed darker for contrast)
    "#46F0F0",  # 7. Bright Electric Cyan (Striking, vibrant cyan)
    "#F032E6",  # 8. Magenta Pink (Now completely distinct from Red #1)
    "#3288BD"   # 9. Steel Blue / Slate Blue (Replacing duplicate green for variety)
]


class EmbeddingsView(QWidget):
    def __init__(self, rvm: RepositoryViewModel, parent=None):
        super().__init__(parent)
        self.rvm = rvm
        self._cached_proj: Optional[np.ndarray] = None
        self._dominant_regimes: Optional[np.ndarray] = None

        self._build_ui()
        self._connect()
        self._populate_info()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Top controls: Multi-select Checkboxes
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Visualize Regimes:"))

        # Master "All" Checkbox
        self.chk_all = QCheckBox("All")
        self.chk_all.setChecked(True)
        ctrl.addWidget(self.chk_all)

        # Individual Checkboxes
        self.regime_checkboxes: list[QCheckBox] = []
        max_dim = getattr(self.rvm, "latent_dim", 9)
        for i in range(1, max_dim + 1):
            chk = QCheckBox(f"Regime {i}")
            chk.setChecked(True)  # Default all on
            ctrl.addWidget(chk)
            self.regime_checkboxes.append(chk)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Plot area (Strictly 3D Plotly WebEngine)
        self.web_view = QWebEngineView()
        self.web_view.setMinimumWidth(450)
        splitter.addWidget(self.web_view)

        # Info panel
        info_frame = QFrame()
        info_frame.setObjectName("card")
        info_l = QVBoxLayout(info_frame)

        info_l.addWidget(QLabel("Model Parameters"))
        self.txt_params = QTextEdit()
        self.txt_params.setReadOnly(True)
        self.txt_params.setMaximumHeight(160)
        info_l.addWidget(self.txt_params)

        info_l.addWidget(QLabel("Degrees of Freedom per Regime"))
        self.txt_dofs = QTextEdit()
        self.txt_dofs.setReadOnly(True)
        self.txt_dofs.setMaximumHeight(180)
        info_l.addWidget(self.txt_dofs)

        info_l.addWidget(QLabel("Assets / Horizons"))
        self.txt_meta = QTextEdit()
        self.txt_meta.setReadOnly(True)
        info_l.addWidget(self.txt_meta)

        splitter.addWidget(info_frame)
        splitter.setSizes([600, 300])
        layout.addWidget(splitter)

        self.status = QLabel("")
        self.status.setObjectName("subtitle")
        layout.addWidget(self.status)

    def _connect(self):
        # Connect master toggles
        self.chk_all.toggled.connect(self._on_all_toggled)
        for chk in self.regime_checkboxes:
            chk.toggled.connect(self._on_regime_toggled)

        self.rvm.data_ready.connect(self._populate_info)

    def _on_all_toggled(self, checked: bool):
        # Programmatically match all children states safely
        for chk in self.regime_checkboxes:
            chk.blockSignals(True)
            chk.setChecked(checked)
            chk.blockSignals(False)
        self._render_plot()

    def _on_regime_toggled(self):
        # Sync the "All" checkbox state based on individual boxes
        all_checked = all(chk.isChecked() for chk in self.regime_checkboxes)
        self.chk_all.blockSignals(True)
        self.chk_all.setChecked(all_checked)
        self.chk_all.blockSignals(False)
        self._render_plot()

    def _populate_info(self):
        params = self.rvm.model_params
        lines = [f"{k}: {v}" for k, v in params.items()]
        self.txt_params.setPlainText("\n".join(lines) or "N/A")

        dof_lines = [f"Regime {k}: ν = {v:.3f}" for k, v in sorted(self.rvm.dofs.items())]
        self.txt_dofs.setPlainText("\n".join(dof_lines))

        meta = (
            f"Assets: {', '.join(self.rvm.assets)}\n"
            f"Horizons: {', '.join(self.rvm.horizons)}\n"
            f"Feature columns: {len(self.rvm.feature_columns)}\n"
            f"Embeddings shape: {None if self.rvm.embeddings is None else self.rvm.embeddings.shape}"
        )
        self.txt_meta.setPlainText(meta)

        # Compute and cache UMAP 3D Coordinates once
        if self.rvm.embeddings is not None and len(self.rvm.embeddings) > 0:
            self.status.setText("Computing 3D UMAP projection (k=9)...")
            try:
                self._cached_proj = self.rvm.get_umap_projection_3d()
                self._dominant_regimes = np.argmax(self.rvm.embeddings, axis=1) + 1
            except Exception as e:
                print(f"Error caching projection: {e}")
                self.status.setText(f"UMAP computation failed: {e}")
                return

        self._render_plot()

    def _render_plot(self):
        if self._cached_proj is None or self._dominant_regimes is None:
            self.status.setText("No projection data available.")
            return

        # Determine which regimes are checked
        selected_regimes = [i + 1 for i, chk in enumerate(self.regime_checkboxes) if chk.isChecked()]

        if not selected_regimes:
            # Clear layout if nothing is selected
            self.web_view.setHtml("<html><body style='background-color:#0f0f0f;color:#e0e0e0;font-family:sans-serif;display:flex;justify-content:center;align-items:center;height:100vh;'><h3>No regimes selected. Check a regime above to visualize.</h3></body></html>")
            self.status.setText("No regimes selected.")
            return

        self.status.setText("Filtering and rendering active regimes...")

        try:
            fig = go.Figure()

            # Mode 1: Exactly one regime selected -> Intensity/Continuous scale
            if len(selected_regimes) == 1:
                regime_idx = selected_regimes[0]
                mask = self._dominant_regimes == regime_idx
                proj_filtered = self._cached_proj[mask]

                if len(proj_filtered) == 0:
                    self.status.setText(f"No points categorized under Regime {regime_idx}.")
                    return

                raw_embeddings = self.rvm.embeddings
                if raw_embeddings is not None:
                    plot_colors = raw_embeddings[mask, regime_idx - 1]
                else:
                    plot_colors = np.ones(len(proj_filtered)) * regime_idx

                fig.add_trace(go.Scatter3d(
                    x=proj_filtered[:, 0],
                    y=proj_filtered[:, 1],
                    z=proj_filtered[:, 2],
                    mode='markers',
                    name=f"Regime {regime_idx}",
                    marker=dict(
                        size=4,
                        opacity=0.85,
                        color=plot_colors,
                        colorscale="Viridis",
                        colorbar=dict(
                            title=f"Regime {regime_idx} Intensity",
                            thickness=15
                        ),
                        showscale=True
                    ),
                    text=[f"Point {i}" for i in np.where(mask)[0]]
                ))
                title_suffix = f"Regime {regime_idx} (Intensity Map)"

            # Mode 2: Multiple regimes selected -> Multi-Trace Categorical legend
            else:
                for r in selected_regimes:
                    mask = self._dominant_regimes == r
                    proj_filtered = self._cached_proj[mask]

                    if len(proj_filtered) == 0:
                        continue

                    # Retrieve consistent discrete color for this regime
                    color_hex = REGIME_COLORS[(r - 1) % len(REGIME_COLORS)]

                    fig.add_trace(go.Scatter3d(
                        x=proj_filtered[:, 0],
                        y=proj_filtered[:, 1],
                        z=proj_filtered[:, 2],
                        mode='markers',
                        name=f"Regime {r}",
                        marker=dict(
                            size=4,
                            opacity=0.85,
                            color=color_hex
                        ),
                        text=[f"Point {i} (R{r})" for i in np.where(mask)[0]]
                    ))
                title_suffix = "Selected Regimes Categorical Comparison"

            fig.update_layout(
                title=f"Latent Embeddings — UMAP 3D ({title_suffix})",
                scene=dict(
                    xaxis_title='UMAP Comp 1',
                    yaxis_title='UMAP Comp 2',
                    zaxis_title='UMAP Comp 3'
                ),
                paper_bgcolor='#0f0f0f',
                plot_bgcolor='#1a1a1a',
                font=dict(color='#e0e0e0'),
                margin=dict(l=0, r=0, b=0, t=40),
                showlegend=True,
                legend=dict(
                    font=dict(color='#e0e0e0'),
                    bgcolor='rgba(0,0,0,0)'
                )
            )

            html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
            self.web_view.setHtml(html)
            self.status.setText(f"Rendering complete.")

        except Exception as e:
            import traceback
            self.status.setText(f"Render error: {e}")
            print(traceback.format_exc())