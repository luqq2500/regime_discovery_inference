import numpy as np
import pandas as pd
import torch
from scipy.stats import t

from core.repository import AssetRepository
from core.usecase.dto import LatentTraversalRequest, LatentTraversalResponse, LatentTraversalOutput, describe_index

class LatentTraversalUseCase:
    def __init__(self, asset_repository: AssetRepository):
        self.asset_repo  = asset_repository
        self.sweep_column = 'sweep'

    def execute(self, request: LatentTraversalRequest) -> LatentTraversalResponse:
        model = self.asset_repo.get_model()
        sweep_values = self._calculate_sweep_values(request.sigma_range)
        traversal_outputs = model(sweep_values)

        outputs: list[LatentTraversalOutput] = []
        for dim, result in traversal_outputs.items():
            sweeps, recon = result
            sweep = self._dataframe_output(sweeps, recon)
            stats = sweep.describe().T
            stats_filtered = request.filter.do_filter(stats)
            sweep_r = self._rescale_output(sweep)
            stats_r = self._rescale_output(stats)
            stats_filtered_r = self._rescale_output(stats_filtered)
            outputs.append(LatentTraversalOutput(dim, sweep, stats, stats_filtered, sweep_r, stats_r, stats_filtered_r))

        return LatentTraversalResponse(outputs)

    def _calculate_sweep_values(self, sigma_range: tuple[float, float], step_multiplier: int=5):
        dof = self.asset_repo.get_degree_of_freedom()
        range_width = abs(sigma_range[1] - sigma_range[0])
        steps = max(5, int(step_multiplier * range_width))
        p_left = t.cdf(sigma_range[0], dof)
        p_right = t.cdf(sigma_range[1], dof)
        probs = np.linspace(p_left, p_right, steps)
        return torch.tensor(t.ppf(probs, df=dof), dtype=torch.float32)

    def _dataframe_output(self, sweep, recon):
        df = pd.DataFrame(recon, columns=self.asset_repo.get_feature_columns())
        df.insert(0, self.sweep_column, sweep)
        return df

    def _rescale_output(self, output: pd.DataFrame) -> pd.DataFrame:
        if output.empty:
            return output

        output = output.copy()

        is_stats = False
        if output.columns.isin(describe_index).any():
            output = output.T
            is_stats = True

        if self.sweep_column in output.columns:
            sweep = output[[self.sweep_column]]
            features = output.drop(columns=[self.sweep_column])
        else:
            sweep = pd.DataFrame(index=output.index)
            features = output

        if features.empty:
            return output.T if is_stats else output

        scaler = self.asset_repo.get_scaler()
        asset_feature_columns = list(self.asset_repo.get_feature_columns())
        feature_columns = list(features.columns)

        scaler_means = np.array([scaler.mean_[asset_feature_columns.index(col)] for col in feature_columns])
        scaler_scales = np.array([scaler.scale_[asset_feature_columns.index(col)] for col in feature_columns])

        if is_stats:
            rescaled_matrix = features.values.astype(float).copy()
            location_rows = ['mean', 'min', '25%', '50%', '75%', 'max']
            for row_name in location_rows:
                if row_name in features.index:
                    row_idx = features.index.get_loc(row_name)
                    raw_log = (rescaled_matrix[row_idx] * scaler_scales) + scaler_means
                    rescaled_matrix[row_idx] = (np.exp(raw_log) - 1.0) * 100.0
            if 'std' in features.index:
                std_idx = features.index.get_loc('std')
                rescaled_matrix[std_idx] = rescaled_matrix[std_idx] * scaler_scales * 100.0
            rescaled_df = pd.DataFrame(rescaled_matrix, columns=features.columns, index=features.index)
        else:
            raw_log = (features.values * scaler_scales) + scaler_means
            percentage_changes = (np.exp(raw_log) - 1.0) * 100.0
            rescaled_df = pd.DataFrame(percentage_changes, columns=features.columns, index=features.index)

        output_df = pd.concat([sweep, rescaled_df], axis=1)

        if is_stats:
            output_df = output_df.T

        return output_df