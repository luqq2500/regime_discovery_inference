import numpy as np
import pandas as pd
import torch
from langchain_core import outputs
from scipy.stats import t

from core.repository import InferenceRepository
from core.usecase.dto import LatentTraversalResponse, LatentTraversalOutput, describe_index, LTARequest


class LatentTraversalUseCase:
    def __init__(self, repository: InferenceRepository):
        self.repository  = repository
        self.sweep_col_name = 'sweep'

    def execute(self, request: LTARequest) -> LatentTraversalResponse:
        model = self.repository.get_decoder()
        outputs: list[LatentTraversalOutput] = []
        for config in request.inputs:
            dof = self.repository.get_degree_of_freedom(dimension=config.dimension)
            target_sweep = self._calculate_sweep_values(sigma_range=config.sigma_range, dof=dof)
            traversal_sweep = self._prepare_traversal_sweeps(target_sweep, config.dimension)
            recon = model(traversal_sweep).detach().numpy()

            sweep_df = self._dataframe_output(target_sweep, recon)
            stats = sweep_df.describe().T
            stats_filtered = config.filter.do_filter(stats)
            sweep_r = self._rescale_output(sweep_df)
            stats_r = self._rescale_output(stats)
            stats_filtered_r = self._rescale_output(stats_filtered)

            print(sweep_df)
            print(stats)
            print(stats_filtered)#
            print(sweep_r)
            print(stats_r)
            print(stats_filtered_r)#


            outputs.append(LatentTraversalOutput(config.dimension, dof, sweep_df, stats, stats_filtered, sweep_r, stats_r, stats_filtered_r))

        '''   
        sweep_values = self._calculate_sweep_values(request.sigma_range)
        traversal_outputs = model(sweep_values)

        outputs: list[LatentTraversalOutput] = []
        for dim, result in traversal_outputs.items():
            traversal_sweep, recon = result
            sweep_df = self._dataframe_output(traversal_sweep, recon)
            stats = sweep_df.describe().T
            stats_filtered = request.filter.do_filter(stats)
            sweep_r = self._rescale_output(sweep_df)
            stats_r = self._rescale_output(stats)
            stats_filtered_r = self._rescale_output(stats_filtered)
            outputs.append(LatentTraversalOutput(dim, sweep_df, stats, stats_filtered, sweep_r, stats_r, stats_filtered_r))
        '''
        return LatentTraversalResponse(outputs)


    def _prepare_traversal_sweeps(self, target_sweep, dimension):
        dims = self.repository.get_dimensions()
        sweeps = torch.zeros((len(target_sweep), dims), device=target_sweep.device)
        sweeps[:, dimension-1] = target_sweep
        return sweeps

    def _calculate_sweep_values(self, sigma_range: tuple[float, float], dof: float, step_multiplier: int=5):
        range_width = abs(sigma_range[1] - sigma_range[0])
        steps = max(5, int(step_multiplier * range_width))
        p_left = t.cdf(sigma_range[0], dof)
        p_right = t.cdf(sigma_range[1], dof)
        probs = np.linspace(p_left, p_right, steps)
        return torch.tensor(t.ppf(probs, df=dof), dtype=torch.float32)

    def _dataframe_output(self, sweep, recon):
        df = pd.DataFrame(recon, columns=self.repository.get_feature_columns())
        df.insert(0, self.sweep_col_name, sweep)
        return df

    def _rescale_output(self, output: pd.DataFrame) -> pd.DataFrame:
        if output.empty:
            return output

        output = output.copy()

        is_stats = False
        if output.columns.isin(describe_index).any():
            output = output.T
            is_stats = True

        if self.sweep_col_name in output.columns:
            sweep = output[[self.sweep_col_name]]
            features = output.drop(columns=[self.sweep_col_name])
        else:
            sweep = pd.DataFrame(index=output.index)
            features = output

        if features.empty:
            return output.T if is_stats else output

        scaler = self.repository.get_scaler()
        asset_feature_columns = list(self.repository.get_feature_columns())
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
            output_df = rescaled_df.T
            if 'snr' in output.columns:
                output_df['snr'] = output_df['mean']/output_df['std'].replace(0, np.nan)
        else:
            raw_log = (features.values * scaler_scales) + scaler_means
            percentage_changes = (np.exp(raw_log) - 1.0) * 100.0
            rescaled_df = pd.DataFrame(percentage_changes, columns=features.columns, index=features.index)
            output_df = pd.concat([sweep, rescaled_df], axis=1)

        return output_df