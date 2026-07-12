import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import t
from sklearn.preprocessing import StandardScaler

from core.repository import InferenceRepository
from core.dto import LatentTraversalInput, LatentTraversalAnalysisOutput, df_describe_index, LatentTraversalOutput


class ModelService:
    def __init__(self, path='assets', encoder_file='encoder.pt', decoder_file='decoder.pt'):
        try:
            self.encoder = torch.jit.load(os.path.join(path, encoder_file), map_location=torch.device('cpu'))
            self.decoder = torch.jit.load(os.path.join(path, decoder_file), map_location=torch.device('cpu'))
        except Exception as e:
            raise RuntimeError(f'Model service initialization error: {e}')

    def encode(self, x)->np.ndarray:
        tensor = self._prepare_tensor(x)
        try:
            torch.inference_mode()
            output = self.encoder(tensor)
            _, _, zs = output
        except Exception as e:
            raise RuntimeError(f'Model service encode error: {e}')
        return zs.detach().numpy()

    def decode(self, z)->np.ndarray:
        tensor = self._prepare_tensor(z)
        try:
            torch.inference_mode()
            output = self.decoder(tensor)
        except Exception as e:
            raise RuntimeError(f'Model service decode error: {e}')
        return output.detach().numpy()

    def _prepare_tensor(self, x):
        if isinstance(x, pd.DataFrame):
            tensor = torch.tensor(x.to_numpy(), dtype=torch.float32)
        elif isinstance(x, np.ndarray):
            tensor = torch.tensor(x, dtype=torch.float32)
        else:
            raise ValueError(f'Invalid input type: {type(x)}')
        return tensor

class ScalerService:
    def __init__(self, path='assets', file_name='asset.pt'):
        try:
            asset = torch.load(os.path.join(path, file_name), map_location=torch.device('cpu'), weights_only=False)
            scaler_type = asset['scaler_type']
            scaler_weights = asset['scaler_weights']
            if scaler_type == 'StandardScaler':
                scaler = StandardScaler()
                scaler.mean_ = scaler_weights['mean']
                scaler.scale_ = scaler_weights['scale']
                scaler.var_ = scaler_weights['var']
                scaler.n_samples_seen_ = scaler_weights['n_samples_seen_']
            else:
                raise ValueError(f'Unsupported scaler type: {scaler_type}')
            self.scaler = scaler
            self.feature_columns = asset['data'].columns.tolist()
        except Exception as e:
            raise RuntimeError(f'Scaler service initialization error: {e}')

    def inverse_transform(self, x: Any) -> np.ndarray:
        return self.scaler.inverse_transform(x)

    def inverse_transform_df(self, features: pd.DataFrame) -> pd.DataFrame:
        try:
            transformed = self.inverse_transform(features)
            return pd.DataFrame(transformed, columns=features.columns, index=features.index)
        except Exception as e:
            raise RuntimeError(f'Scaler service method inverse_transform_df error: {e}')

    def inverse_transform_df_stats(self, features: pd.DataFrame) -> pd.DataFrame:
        means, scales = self._get_scales_for(list(features.columns))
        matrix = features.values.astype(float).copy()
        location_rows = ['mean', 'min', '25%', '50%', '75%', 'max']
        for row_name in location_rows:
            if row_name in features.index:
                idx = features.index.get_loc(row_name)
                matrix[idx] = self._inverse_transform_values(matrix[idx], means, scales)
        if 'std' in features.index:
            idx = features.index.get_loc('std')
            matrix[idx] = self._inverse_transform_std(matrix[idx], scales)
        return pd.DataFrame(matrix, columns=features.columns, index=features.index)

    def _get_scales_for(self, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
        means = np.array([self.scaler.mean_[self.feature_columns.index(col)] for col in feature_columns])
        scales = np.array([self.scaler.scale_[self.feature_columns.index(col)] for col in feature_columns])
        return means, scales

    def _inverse_transform_values(self, values: np.ndarray, means: np.ndarray, scales: np.ndarray) -> np.ndarray:
        raw_log = (values * scales) + means
        return (np.exp(raw_log) - 1.0) * 100.0

    def _inverse_transform_std(self, values: np.ndarray, scales: np.ndarray) -> np.ndarray:
        return values * scales * 100.0

class LatentTraversalService:
    def __init__(self, model_service: ModelService, repository: InferenceRepository):
        self.model_service = model_service
        self.repository = repository

    def execute(self, lt_input: LatentTraversalInput) -> LatentTraversalOutput:
        dof = self.repository.get_dof_by_dimension(dimension=lt_input.dimension - 1)
        target_sweep = self._calculate_sweep_values(sigma_range=lt_input.sigma_range, dof=dof)
        traversal_sweep = self._prepare_traversal_sweeps(target_sweep, lt_input.dimension)
        recon = self.model_service.decode(traversal_sweep)
        return LatentTraversalOutput(lt_input.dimension, round(dof, 2), target_sweep.tolist(), recon)

    def _prepare_traversal_sweeps(self, target_sweep, dimension):
        dims = self.repository.get_dimensions()
        sweeps = np.zeros((len(target_sweep), dims))
        sweeps[:, dimension - 1] = target_sweep
        return sweeps

    def _calculate_sweep_values(self, sigma_range: tuple[float, float], dof: float, step_multiplier: int = 5)->np.ndarray:
        range_width = abs(sigma_range[1] - sigma_range[0])
        steps = max(5, int(step_multiplier * range_width))
        p_left = t.cdf(sigma_range[0], dof)
        p_right = t.cdf(sigma_range[1], dof)
        probs = np.linspace(p_left, p_right, steps)
        return t.ppf(probs, df=dof)

