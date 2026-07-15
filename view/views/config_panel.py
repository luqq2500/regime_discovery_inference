"""
Configuration sidebar panel.
Pure View: builds widgets and forwards user actions to ViewModel.
"""
from __future__ import annotations


from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QSpinBox, QGroupBox,
    QFrame,QFormLayout, QButtonGroup, QGridLayout
)

from view.viewmodels.analysis_vm import AnalysisViewModel


class RegimeCard(QFrame):
    """An ultra-compact, clickable card representing a single analysis regime."""
    clicked = pyqtSignal(int, bool)  # dimension, selected state

    def __init__(self, dimension: int, dof: float, is_selected: bool = False, parent=None):
        super().__init__(parent)
        self.dimension = dimension
        self.dof = dof
        self._selected = is_selected
        self.setObjectName("regimeCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Compact height bounds to allow 3 rows to fit easily without overlapping other sections
        self.setMinimumHeight(34)
        self.setMaximumHeight(38)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Compact "RX" label
        self.lbl_title = QLabel(f"R{dimension} ({dof:.1f})")
        self.lbl_title.setStyleSheet("font-weight: bold; font-size: 10px; color: #ffffff;")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_title)

        '''
        self.lbl_dof = QLabel(f"ν≈{dof:.1f}")
        self.lbl_dof.setStyleSheet("font-size: 8px; color: #8a8a8a;")
        self.lbl_dof.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_dof)
        '''


        self._update_style()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._selected = not self._selected
            self._update_style()
            self.clicked.emit(self.dimension, self._selected)

    def set_selected(self, selected: bool, block_signal: bool = False):
        if self._selected != selected:
            self._selected = selected
            self._update_style()
            if not block_signal:
                self.clicked.emit(self.dimension, self._selected)

    def is_selected(self) -> bool:
        return self._selected

    def _update_style(self):
        if self._selected:
            self.setStyleSheet("""
                QFrame#regimeCard {
                    border: 1.5px solid #00b4ff; 
                    background-color: #12283a;
                    border-radius: 4px;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame#regimeCard {
                    border: 1px solid #333333; 
                    background-color: #1a1a1a;
                    border-radius: 4px;
                }
                QFrame#regimeCard:hover {
                    border: 1px solid #555555;
                    background-color: #222222;
                }
            """)


class ConfigPanel(QWidget):
    """Left sidebar for input configuration."""

    run_requested = pyqtSignal()

    def __init__(self, vm: AnalysisViewModel, parent=None):
        super().__init__(parent)
        self.vm = vm
        self.setObjectName("configPanel")
        self.setMinimumWidth(280)
        self.setMaximumWidth(340)
        self.regime_cards: dict[int, RegimeCard] = {}
        self._build_ui()
        self._connect()
        self._sync_from_vm()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # Title
        title = QLabel("Regime Discovery")
        title.setObjectName("title")
        root.addWidget(title)

        sub = QLabel("Latent Traversal Analysis")
        sub.setObjectName("subtitle")
        root.addWidget(sub)

        # --- Regime Selection (Compact 3x3 Layout Grid) ---
        g_regimes = QGroupBox("Regimes (Degree of Freedom)")

        gl = QVBoxLayout(g_regimes)
        gl.setSpacing(6)
        gl.setContentsMargins(6, 8, 6, 8)

        grid = QGridLayout()
        grid.setSpacing(4)

        self.regime_cards.clear()
        for d in self.vm.available_dimensions:
            dof = self.vm.get_dof(d)
            is_sel = d in self.vm.state.dimensions
            card = RegimeCard(d, dof, is_selected=is_sel)
            card.clicked.connect(self._on_regime_card_clicked)

            # Map index seamlessly to a perfect 3x3 layout grid
            row = (d - 1) // 3
            col = (d - 1) % 3
            grid.addWidget(card, row, col)

            self.regime_cards[d] = card

        gl.addLayout(grid)

        # Quick action buttons
        btn_row = QHBoxLayout()
        self.btn_all = QPushButton("All")
        self.btn_clear = QPushButton("Clear")
        btn_row.addWidget(self.btn_all)
        btn_row.addWidget(self.btn_clear)
        gl.addLayout(btn_row)

        root.addWidget(g_regimes)

        # --- Sigma Range ---
        g_sigma = QGroupBox("Market Condition (+-σ Range)")
        form = QFormLayout(g_sigma)

        self.sigma_min = QDoubleSpinBox()
        self.sigma_min.setRange(-50.0, 50.0)
        self.sigma_min.setSingleStep(0.1)
        self.sigma_min.setDecimals(2)
        self.sigma_min.setValue(self.vm.state.sigma_min)

        self.sigma_max = QDoubleSpinBox()
        self.sigma_max.setRange(-50.0, 50.0)
        self.sigma_max.setSingleStep(0.1)
        self.sigma_max.setDecimals(2)
        self.sigma_max.setValue(self.vm.state.sigma_max)

        form.addRow("σ Min", self.sigma_min)
        form.addRow("σ Max", self.sigma_max)
        root.addWidget(g_sigma)

        # --- Filter ---
        g_filter = QGroupBox("Asset Filter")
        form2 = QFormLayout(g_filter)

        self.top_n = QSpinBox()
        self.top_n.setRange(1, 50)
        self.top_n.setValue(self.vm.state.top_n)

        self.snr = QDoubleSpinBox()
        self.snr.setRange(0.0, 100.0)
        self.snr.setSingleStep(0.05)
        self.snr.setDecimals(2)
        self.snr.setValue(self.vm.state.snr_threshold)

        form2.addRow("Top N", self.top_n)
        form2.addRow("SNR ≥", self.snr)
        root.addWidget(g_filter)

        # --- Display Toggle Buttons ---
        g_disp = QGroupBox("Display")
        dl = QHBoxLayout(g_disp)
        dl.setSpacing(6)
        dl.setContentsMargins(6, 8, 6, 8)

        # Local stylesheet to visually highlight the active toggle state
        toggle_style = """
            QPushButton {
                background-color: #1a1a1a;
                color: #a0a0a0;
                border: 1px solid #333333;
                border-radius: 4px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #222222;
                border: 1px solid #555555;
            }
            QPushButton:checked {
                background-color: #12283a;
                color: #ffffff;
                border: 1.5px solid #00b4ff;
            }
        """

        self.btn_raw = QPushButton("Raw %")
        self.btn_raw.setCheckable(True)
        self.btn_raw.setMinimumHeight(28)
        self.btn_raw.setStyleSheet(toggle_style)

        self.btn_scaled = QPushButton("Scaled")
        self.btn_scaled.setCheckable(True)
        self.btn_scaled.setMinimumHeight(28)
        self.btn_scaled.setStyleSheet(toggle_style)

        self.disp_group = QButtonGroup(self)
        self.disp_group.setExclusive(True)
        self.disp_group.addButton(self.btn_raw)
        self.disp_group.addButton(self.btn_scaled)

        is_raw = self.vm.state.use_raw
        self.btn_raw.setChecked(is_raw)
        self.btn_scaled.setChecked(not is_raw)

        dl.addWidget(self.btn_raw)
        dl.addWidget(self.btn_scaled)
        root.addWidget(g_disp)

        # --- Run ---
        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.setObjectName("primaryButton")
        self.btn_run.setMinimumHeight(40)
        root.addWidget(self.btn_run)

        self.status = QLabel("Ready")
        self.status.setObjectName("subtitle")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        root.addStretch()

        # Footer info
        info = QLabel(f"Latent dim: {self.vm.max_dimension}  |  Features: {len(self.vm.feature_columns)}")
        info.setObjectName("subtitle")
        root.addWidget(info)

    def _connect(self):
        self.btn_run.clicked.connect(self._on_run)
        self.btn_all.clicked.connect(self._select_all)
        self.btn_clear.clicked.connect(self._clear)
        self.sigma_min.valueChanged.connect(self._on_sigma)
        self.sigma_max.valueChanged.connect(self._on_sigma)
        self.top_n.valueChanged.connect(lambda v: self.vm.set_top_n(v))
        self.snr.valueChanged.connect(lambda v: self.vm.set_snr_threshold(v))

        self.btn_raw.clicked.connect(lambda: self.vm.set_use_raw(True))
        self.btn_scaled.clicked.connect(lambda: self.vm.set_use_raw(False))

        self.vm.state_changed.connect(self._sync_from_vm)
        self.vm.analysis_started.connect(lambda: self._set_busy(True))
        self.vm.analysis_finished.connect(lambda _: self._set_busy(False))
        self.vm.analysis_failed.connect(lambda e: self._set_busy(False, e))
        self.vm.progress.connect(self.status.setText)

    def _on_run(self):
        self.run_requested.emit()
        self.vm.run_analysis()

    def _select_all(self):
        for card in self.regime_cards.values():
            card.set_selected(True, block_signal=True)
        dims = list(self.regime_cards.keys())
        self.vm.set_dimensions(dims)

    def _clear(self):
        for card in self.regime_cards.values():
            card.set_selected(False, block_signal=True)
        self.vm.set_dimensions([])

    def _on_regime_card_clicked(self, dimension: int, selected: bool):
        dims = [d for d, card in self.regime_cards.items() if card.is_selected()]
        self.vm.set_dimensions(dims)

    def _on_sigma(self):
        self.vm.set_sigma_range(self.sigma_min.value(), self.sigma_max.value())

    def _set_busy(self, busy: bool, err: str = None):
        self.btn_run.setEnabled(not busy)
        if busy:
            self.status.setText("Running...")
        elif err:
            self.status.setText(f"Error: {err[:80]}")
        else:
            self.status.setText("Done")

    def _sync_from_vm(self):
        # Sync Display Toggle Buttons
        self.btn_raw.blockSignals(True)
        self.btn_scaled.blockSignals(True)
        is_raw = self.vm.state.use_raw
        self.btn_raw.setChecked(is_raw)
        self.btn_scaled.setChecked(not is_raw)
        self.btn_raw.blockSignals(False)
        self.btn_scaled.blockSignals(False)

        # Sync Regime cards
        for d, card in self.regime_cards.items():
            card.set_selected(d in self.vm.state.dimensions, block_signal=True)