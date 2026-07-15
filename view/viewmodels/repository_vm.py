"""
ViewModel for repository diagnostics and exploration.
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd

from PyQt6.QtCore import QObject, pyqtSignal

from core.repository.repository import InferenceRepository


class RepositoryViewModel(QObject):
    """
    Exposes repository contents for UI diagnostics.
    Single responsibility: transform repo data into UI-ready structures.
    """

    data_ready = pyqtSignal()

    def __init__(self, repository: InferenceRepository, parent=None):
        super().__init__(parent)
        self._repo = repository
        self._embeddings: Optional[np.ndarray] = None
        self._feature: Optional[pd.DataFrame] = None
        self._model_params: Dict[str, Any] = {}
        self._dofs: Dict = {}
        self._load()

    def _load(self):
        try:
            self._embeddings = self._repo.get_embeddings()
            self._feature = self._repo.get_feature()
            self._model_params = self._repo.get_model_params()
            raw_dofs = self._repo.get_dofs()
            if isinstance(raw_dofs, dict):
                self._dofs = {int(k) + 1: float(v) for k, v in raw_dofs.items()}
            else:
                self._dofs = {i + 1: float(v) for i, v in enumerate(raw_dofs)}
            self.data_ready.emit()
        except Exception as e:
            print(f"RepositoryViewModel load error: {e}")

    @property
    def embeddings(self) -> Optional[np.ndarray]:
        return self._embeddings

    @property
    def feature_df(self) -> Optional[pd.DataFrame]:
        return self._feature

    @property
    def model_params(self) -> Dict[str, Any]:
        return self._model_params

    @property
    def dofs(self) -> Dict[int, float]:
        return self._dofs

    @property
    def latent_dim(self) -> int:
        return self._model_params.get("latent_dim", 9)

    @property
    def input_dim(self) -> int:
        return self._model_params.get("input_dim", 25)

    @property
    def nu(self) -> float:
        return self._model_params.get("nu", 5.0)

    @property
    def assets(self) -> List[str]:
        return self._repo.get_feature_assets()

    @property
    def horizons(self) -> List[str]:
        return self._repo.get_feature_horizons()

    @property
    def feature_columns(self) -> List[str]:
        return list(self._repo.get_feature_columns())

    def get_umap_projection_3d(self) -> Optional[np.ndarray]:
        """
        Saves and returns the 9D UMAP projection mapped down to 3D.
        Uses Euclidean metrics and is safe against NaNs/Infs.
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return None

        # Defensive clean against NaNs/Infs
        X = np.nan_to_num(self._embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        import umap
        # Project down to 3 components for 3D rendering
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
            metric="euclidean"
        )
        return reducer.fit_transform(X)

    def get_feature_summary(self) -> pd.DataFrame:
        if self._feature is None:
            return pd.DataFrame()
        return self._feature.describe().T