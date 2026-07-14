import numpy as np
import pandas as pd

from core.domain import df_describe_index
from core.usecase.dto import LatentTraversalAnalysisResponse, LatentTraversalAnalysisOutput, LatentTraversalAnalysisRequest
from core.repository.repository import InferenceRepository
from core.service.service import LatentTraversalService, ScalerService
from core.usecase.interface import LatentTraversalAnalyserUseCase


class LatentTraversalAnalysisUseCase(LatentTraversalAnalyserUseCase):
    def __init__(self, traversal_service: LatentTraversalService,
                 scaler_service: ScalerService,
                 inference_repository: InferenceRepository):

        self.traversal_service = traversal_service
        self.scaler_service = scaler_service
        self.repository = inference_repository
        self.traversal_col_name = 'sweep'

    def analyse(self, request: LatentTraversalAnalysisRequest) -> LatentTraversalAnalysisResponse:
        results: list[LatentTraversalAnalysisOutput] = []
        lt_inputs = request.inputs
        for input_config in lt_inputs:
            result = self.traversal_service.execute(input_config)

            sweep = self._dataframe_output(result.sweeps, result.recon)
            stats = sweep.describe().T
            stats_filtered = input_config.filter.do_filter(stats)

            sweep_r = self._rescale(sweep)
            stats_r = self._rescale(stats.T)
            stats_filtered_r = self._rescale(stats_filtered.T)

            results.append(LatentTraversalAnalysisOutput(
                regime=input_config.dimension,
                degree_of_freedom=result.degree_of_freedom,
                traversal_sweeps=result.sweeps,
                result_scaled=sweep,
                asset_stats_scaled=stats,
                filtered_asset_stats_scaled=stats_filtered,
                result_raw=sweep_r,
                asset_stats_raw=stats_r,
                filtered_asset_stats_raw=stats_filtered_r,
            ))
        return LatentTraversalAnalysisResponse(results)

    def _dataframe_output(self, sweep, recon):
        df = pd.DataFrame(recon, columns=self.repository.get_feature_columns())
        df.insert(0, self.traversal_col_name, sweep)
        return df

    def _rescale(self, result: pd.DataFrame) -> pd.DataFrame:
        result = result.copy()

        is_describe = result.index.isin(df_describe_index).any()
        has_snr = 'snr' in result.index
        has_sweep = self.traversal_col_name in result.columns
        sweep = result[[self.traversal_col_name]] if has_sweep else pd.DataFrame(index=result.index)
        features = result.drop(columns=[self.traversal_col_name]) if has_sweep else result

        if features.empty:
            return result.T if is_describe else result

        if is_describe:
            rescaled_df = self.scaler_service.inverse_transform_df_describe(features)
            output_df = rescaled_df.T
            if has_snr:
                output_df['snr'] = abs(output_df['mean'] / output_df['std'].replace(0, np.nan))
        else:
            rescaled_df = self.scaler_service.inverse_transform_df(features)
            output_df = pd.concat([sweep, rescaled_df], axis=1)

        return output_df
