"""
Repository explorer: embeddings projection + model / DOF info.
"""
from __future__ import annotations

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QTextEdit, QSplitter, QFrame, QFormLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtWebEngineWidgets import QWebEngineView
import pyqtgraph as pg
import plotly.graph_objects as go
import plotly.io as pio

from view.viewmodels.repository_vm import RepositoryViewModel


class EmbeddingsView(QWidget):
    def __init__(self, rvm: RepositoryViewModel, parent=None):
        super().__init__(parent)
        self.rvm = rvm
        self._build_ui()
        self._connect()
        self._populate_info()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Top controls
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Projection:"))
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(["PCA", "t-SNE", "UMAP"])
        # Set default projection method to UMAP
        self.cmb_method.setCurrentText("UMAP")
        ctrl.addWidget(self.cmb_method)

        ctrl.addWidget(QLabel("  n_components:"))
        self.cmb_dims = QComboBox()
        # Dynamically fill 2 .. latent_dim (default 9)
        max_dim = getattr(self.rvm, "latent_dim", 9)
        self.cmb_dims.addItems([str(i) for i in range(2, max_dim + 1)])

        # Select '9' as the default. Fall back to the maximum index if 9 is out of range.
        default_dim_idx = self.cmb_dims.findText("9")
        if default_dim_idx != -1:
            self.cmb_dims.setCurrentIndex(default_dim_idx)
        else:
            self.cmb_dims.setCurrentIndex(self.cmb_dims.count() - 1)

        ctrl.addWidget(self.cmb_dims)

        ctrl.addWidget(QLabel("  Color by:"))
        self.cmb_color = QComboBox()
        self.cmb_color.addItem("Uniform (blue)", None)
        max_dim = getattr(self.rvm, "latent_dim", 9)
        for i in range(1, max_dim + 1):
            self.cmb_color.addItem(f"Latent Dim {i} (Regime {i})", i)
        ctrl.addWidget(self.cmb_color)

        self.btn_recompute = QPushButton("Recompute")
        ctrl.addWidget(self.btn_recompute)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Plot area: switchable 2D pyqtgraph or 3D plotly
        self.plot_container = QWidget()
        plot_layout = QVBoxLayout(self.plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.pg_plot = pg.PlotWidget()
        self.pg_plot.setBackground("#0f0f0f")
        self.pg_plot.showGrid(x=True, y=True, alpha=0.2)
        self.pg_plot.setLabel("bottom", "Component 1")
        self.pg_plot.setLabel("left", "Component 2")
        self.pg_plot.setMinimumWidth(450)
        plot_layout.addWidget(self.pg_plot)

        self.web_view = QWebEngineView()
        self.web_view.setMinimumWidth(450)
        plot_layout.addWidget(self.web_view)
        self.web_view.hide()

        splitter.addWidget(self.plot_container)

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
        self.btn_recompute.clicked.connect(self._draw_projection)
        self.cmb_method.currentTextChanged.connect(self._draw_projection)
        self.cmb_dims.currentTextChanged.connect(self._draw_projection)
        self.cmb_color.currentIndexChanged.connect(self._draw_projection)
        self.rvm.data_ready.connect(self._populate_info)

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
        self._draw_projection()

    def _draw_projection(self):
        method = self.cmb_method.currentText().lower().replace("-", "")
        n_comp = int(self.cmb_dims.currentText())
        color_dim = self.cmb_color.currentData()
        color_label = self.cmb_color.currentText() if color_dim else "Uniform"

        self.status.setText(f"Computing {method.upper()} ({n_comp}D, color={color_label})...")

        try:
            proj = self.rvm.get_embedding_projection(method=method, n_components=n_comp)
            if proj is None:
                self.status.setText("No embeddings available.")
                return

            self.pg_plot.clear()
            self.web_view.hide()
            self.pg_plot.show()

            # Get color values if requested (use original embeddings column; full data used for UMAP/PCA)
            colors = None
            if color_dim is not None and self.rvm.embeddings is not None:
                try:
                    colors = self.rvm.embeddings[:, color_dim - 1]
                    if len(colors) != len(proj):
                        colors = colors[:len(proj)]  # fallback if subsampled
                except Exception:
                    colors = None

            if n_comp >= 3:
                # 3D Plotly of first 3 components (with optional coloring by latent regime dim)
                title_suffix = f"{n_comp}D → first 3 comps" if n_comp > 3 else "3D"
                marker_dict = dict(size=4, opacity=0.85)
                if colors is not None:
                    marker_dict.update(dict(
                        color=colors,
                        colorscale="Viridis",
                        colorbar=dict(title=f"Latent Dim {color_dim}", thickness=15),
                        showscale=True
                    ))
                else:
                    marker_dict.update(dict(color="#00b4ff"))

                fig = go.Figure(data=[go.Scatter3d(
                    x=proj[:, 0],
                    y=proj[:, 1],
                    z=proj[:, 2],
                    mode='markers',
                    marker=marker_dict,
                    text=[f"Point {i}" for i in range(len(proj))] if colors is None else None
                )])
                fig.update_layout(
                    title=f"Latent Embeddings — {method.upper()} ({title_suffix}, colored by {color_label})",
                    scene=dict(
                        xaxis_title='Comp 1',
                        yaxis_title='Comp 2',
                        zaxis_title='Comp 3'
                    ),
                    paper_bgcolor='#0f0f0f',
                    plot_bgcolor='#1a1a1a',
                    font=dict(color='#e0e0e0')
                )
                html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
                self.web_view.setHtml(html)
                self.web_view.show()
                self.pg_plot.hide()
                self.status.setText(
                    f"{method.upper()} {n_comp}D embedding computed "
                    f"({proj.shape[0]} points) — showing first 3 components in 3D"
                    + (f", colored by Latent Dim {color_dim}" if color_dim else "")
                )
            else:
                # 2D pyqtgraph
                if colors is not None:
                    try:
                        from PyQt6.QtGui import QBrush, QColor
                        import matplotlib.cm as cm
                        import matplotlib.colors as mcolors
                        norm = mcolors.Normalize(vmin=np.min(colors), vmax=np.max(colors))
                        cmap = cm.get_cmap('viridis')
                        brushes = []
                        for cval in colors:
                            rgba = cmap(norm(cval))
                            brushes.append(QBrush(QColor(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))))
                        spots = [{"pos": (x, y), "size": 5, "pen": None, "brush": b}
                                 for (x, y), b in zip(proj, brushes)]
                    except Exception:
                        spots = [{"pos": (x, y), "size": 6, "pen": None, "brush": pg.mkBrush("#00b4ff")}
                                 for x, y in proj]
                else:
                    spots = [{"pos": (x, y), "size": 6, "pen": None, "brush": pg.mkBrush("#00b4ff")}
                             for x, y in proj]
                scatter = pg.ScatterPlotItem(spots)
                self.pg_plot.addItem(scatter)
                self.pg_plot.setTitle(f"Latent Embeddings — {method.upper()} (2D, {color_label})", color="#00b4ff",
                                      size="11pt")
                self.status.setText(f"{method.upper()} 2D projection of {proj.shape[0]} points")
        except Exception as e:
            import traceback
            self.status.setText(f"Projection error: {e}")
            print(traceback.format_exc())