import numpy as np
import pandas as pd
from scipy.stats import t
from core.repository import AssetRepository
from core.service import ModelService, ScalerService
from core.usecase.dto import LatentTraversalOutput, df_describe_index, LatentTraversalInput


class LatentTraversalAnalysisService:
    def __init__(self, model_service: ModelService, scaler_service: ScalerService, repository: AssetRepository):
        self.model_service = model_service
        self.scaler_service = scaler_service
        self.repository  = repository
        self.sweep_col_name = 'sweep'

    def execute(self, lt_input: LatentTraversalInput) -> LatentTraversalOutput:
        dof = self.repository.get_dof_by_dimension(dimension=lt_input.dimension-1)
        target_sweep = self._calculate_sweep_values(sigma_range=lt_input.sigma_range, dof=dof)
        traversal_sweep = self._prepare_traversal_sweeps(target_sweep, lt_input.dimension)
        recon = self.model_service.decode(traversal_sweep)
        sweep_df = self._dataframe_output(target_sweep, recon)
        stats = sweep_df.describe().T
        stats_filtered = lt_input.filter.do_filter(stats)
        sweep_r = self._rescale_output(sweep_df)
        stats_r = self._rescale_output(stats.T)
        stats_filtered_r = self._rescale_output(stats_filtered.T)
        return LatentTraversalOutput(lt_input.dimension, dof, target_sweep, sweep_df, stats, stats_filtered, sweep_r, stats_r, stats_filtered_r)

    def _prepare_traversal_sweeps(self, target_sweep, dimension):
        dims = self.repository.get_dimensions()
        sweeps = np.zeros((len(target_sweep), dims))
        sweeps[:, dimension-1] = target_sweep
        return sweeps

    def _calculate_sweep_values(self, sigma_range: tuple[float, float], dof: float, step_multiplier: int=5):
        range_width = abs(sigma_range[1] - sigma_range[0])
        steps = max(5, int(step_multiplier * range_width))
        p_left = t.cdf(sigma_range[0], dof)
        p_right = t.cdf(sigma_range[1], dof)
        probs = np.linspace(p_left, p_right, steps)
        return t.ppf(probs, df=dof)

    def _dataframe_output(self, sweep, recon):
        df = pd.DataFrame(recon, columns=self.repository.get_feature_columns())
        df.insert(0, self.sweep_col_name, sweep)
        return df

    def _rescale_output(self, output: pd.DataFrame) -> pd.DataFrame:
        output = output.copy()
        is_stats = output.index.isin(df_describe_index).any()
        has_snr = 'snr' in output.index
        has_sweep = self.sweep_col_name in output.columns

        sweep = output[[self.sweep_col_name]] if has_sweep else pd.DataFrame(index=output.index)
        features = output.drop(columns=[self.sweep_col_name]) if has_sweep else output
        if features.empty:
            return output.T if is_stats else output

        if is_stats:
            rescaled_df = self.scaler_service.inverse_transform_df_stats(features)
            output_df = rescaled_df.T
            if has_snr:
                output_df['snr'] = output_df['mean'] / output_df['std'].replace(0, np.nan)
        else:
            rescaled_df = self.scaler_service.inverse_transform_df(features)
            output_df = pd.concat([sweep, rescaled_df], axis=1)

        return output_df