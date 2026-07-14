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

    def get_embedding_projection(self, method: str = "pca", n_components: int = 2) -> Optional[np.ndarray]:
        """
        Project embeddings with PCA / t-SNE / UMAP.
        n_components can be up to latent_dim (9).
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return None
        X = self._embeddings.copy()
        n_samples, n_features = X.shape

        # Clamp n_components to valid range
        max_possible = min(n_samples - 1, n_features, 50)  # hard safety cap
        n_components = max(2, min(int(n_components), max_possible))

        # subsample only for t-SNE (slow); UMAP handles full data fine and preserves coloring
        if n_samples > 2000 and method.lower() == "tsne":
            idx = np.random.choice(n_samples, size=2000, replace=False)
            X = X[idx]

        method = method.lower()
        if method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == "tsne":
            from sklearn.manifold import TSNE
            # sklearn TSNE is slow / limited for high n_components; still allow
            reducer = TSNE(
                n_components=n_components,
                perplexity=min(30, X.shape[0] // 4),
                random_state=42,
                n_iter=500,
                init="pca",
            )
        elif method == "umap":
            import umap
            # UMAP happily supports n_components up to n_features (here 9)
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=15,
                min_dist=0.1,
                random_state=42,
                metric="euclidean",
            )
        else:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components, random_state=42)

        return reducer.fit_transform(X)

    def get_feature_summary(self) -> pd.DataFrame:
        if self._feature is None:
            return pd.DataFrame()
        return self._feature.describe().T
