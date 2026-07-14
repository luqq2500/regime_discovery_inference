"""
ViewModel for LatentTraversalAnalysisUseCase.
Single responsibility: UI state, request building, response transformation, and interaction logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from core.usecase.dto import (
    LatentTraversalAnalysisRequest,
    LatentTraversalAnalysisResponse,
    LatentTraversalAnalysisOutput,
)
from core.usecase.usecase import LatentTraversalAnalysisUseCase
from core.repository.repository import InferenceRepository


@dataclass
class UIAnalysisState:
    """Pure data container for UI state (SRP)."""
    is_running: bool = False
    last_error: Optional[str] = None
    dimensions: List[int] = field(default_factory=lambda: [1])
    sigma_min: float = -3.0
    sigma_max: float = 3.0
    top_n: int = 10
    snr_threshold: float = 0.1
    selected_regime: Optional[int] = None
    use_raw: bool = True  # True = show raw %, False = scaled
    response: Optional[LatentTraversalAnalysisResponse] = None


class AnalysisViewModel(QObject):
    """
    MVVM ViewModel.
    - Owns UI state
    - Builds requests from UI controls
    - Calls UseCase
    - Emits signals for Views to react
    - No direct UI widgets
    """

    # Signals for View binding
    state_changed = pyqtSignal()
    analysis_started = pyqtSignal()
    analysis_finished = pyqtSignal(object)  # LatentTraversalAnalysisResponse
    analysis_failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        usecase: LatentTraversalAnalysisUseCase,
        repository: InferenceRepository,
        parent=None,
    ):
        super().__init__(parent)
        self._usecase = usecase
        self._repo = repository
        self._state = UIAnalysisState()

        # Cache meta from repo
        self._max_dim = self._repo.get_dimensions()
        self._dofs = self._repo.get_dofs()
        self._assets = self._repo.get_feature_assets()
        self._horizons = self._repo.get_feature_horizons()
        self._feature_columns = list(self._repo.get_feature_columns())

    # ---------- Public properties (for Views) ----------
    @property
    def state(self) -> UIAnalysisState:
        return self._state

    @property
    def max_dimension(self) -> int:
        return self._max_dim

    @property
    def available_dimensions(self) -> List[int]:
        return list(range(1, self._max_dim + 1))

    @property
    def dofs(self) -> Dict[int, float]:
        # repo returns dict with 0-based keys
        return {k + 1: v for k, v in self._dofs.items()} if isinstance(self._dofs, dict) else {
            i + 1: self._dofs[i] for i in range(len(self._dofs))
        }

    @property
    def assets(self) -> List[str]:
        return self._assets

    @property
    def horizons(self) -> List[str]:
        return self._horizons

    @property
    def feature_columns(self) -> List[str]:
        return self._feature_columns

    # ---------- Setters (called by View) ----------
    def set_dimensions(self, dims: List[int]):
        clean = sorted(set(d for d in dims if 1 <= d <= self._max_dim))
        if not clean:
            clean = [1]
        self._state.dimensions = clean
        self.state_changed.emit()

    def set_sigma_range(self, lo: float, hi: float):
        self._state.sigma_min = min(lo, hi)
        self._state.sigma_max = max(lo, hi)
        self.state_changed.emit()

    def set_top_n(self, n: int):
        self._state.top_n = max(1, min(50, n))
        self.state_changed.emit()

    def set_snr_threshold(self, thr: float):
        self._state.snr_threshold = max(0.0, thr)
        self.state_changed.emit()

    def set_use_raw(self, use_raw: bool):
        self._state.use_raw = use_raw
        self.state_changed.emit()

    def select_regime(self, regime: int):
        self._state.selected_regime = regime
        self.state_changed.emit()

    # ---------- Core action ----------
    @pyqtSlot()
    def run_analysis(self):
        if self._state.is_running:
            return

        self._state.is_running = True
        self._state.last_error = None
        self.analysis_started.emit()
        self.progress.emit("Building request...")
        self.state_changed.emit()

        try:
            # Note: LatentTraversalInput currently hardcodes snr via top_n only.
            # snr_threshold is supported in domain but request currently only exposes top_n.
            request = LatentTraversalAnalysisRequest(
                dimensions=self._state.dimensions,
                sigma_range=(self._state.sigma_min, self._state.sigma_max),
                top_n=self._state.top_n,
            )
            # Manually inject snr if domain supports (future-proof)
            for inp in request.inputs:
                inp.snr_threshold = self._state.snr_threshold
                inp.filter.snr_threshold = self._state.snr_threshold

            self.progress.emit(f"Running latent traversal for regimes {self._state.dimensions}...")
            response = self._usecase.analyse(request)

            self._state.response = response
            if response.outputs:
                self._state.selected_regime = response.outputs[0].regime

            self.progress.emit("Analysis complete.")
            self.analysis_finished.emit(response)
        except Exception as e:
            err = str(e)
            self._state.last_error = err
            self.analysis_failed.emit(err)
            self.progress.emit(f"Error: {err}")
        finally:
            self._state.is_running = False
            self.state_changed.emit()

    # ---------- Helpers for Views (pure transformation) ----------
    def get_output_for_regime(self, regime: int) -> Optional[LatentTraversalAnalysisOutput]:
        if not self._state.response:
            return None
        for out in self._state.response.outputs:
            if out.regime == regime:
                return out
        return None

    def get_current_output(self) -> Optional[LatentTraversalAnalysisOutput]:
        if self._state.selected_regime is None:
            return None
        return self.get_output_for_regime(self._state.selected_regime)

    def get_traversal_df(self, regime: Optional[int] = None) -> Optional[pd.DataFrame]:
        out = self.get_output_for_regime(regime) if regime else self.get_current_output()
        if out is None:
            return None
        return out.result_raw if self._state.use_raw else out.result_scaled

    def get_filtered_stats(self, regime: Optional[int] = None) -> Optional[pd.DataFrame]:
        out = self.get_output_for_regime(regime) if regime else self.get_current_output()
        if out is None:
            return None
        return out.filtered_asset_stats_raw if self._state.use_raw else out.filtered_asset_stats_scaled

    def get_full_stats(self, regime: Optional[int] = None) -> Optional[pd.DataFrame]:
        out = self.get_output_for_regime(regime) if regime else self.get_current_output()
        if out is None:
            return None
        return out.asset_stats_raw if self._state.use_raw else out.asset_stats_scaled

    def get_dof(self, regime: int) -> float:
        return self.dofs.get(regime, 0.0)

    def get_all_outputs(self) -> List[LatentTraversalAnalysisOutput]:
        if not self._state.response:
            return []
        return self._state.response.outputs
