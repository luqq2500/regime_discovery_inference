import numpy as np
import pandas as pd
from core.dto import LatentTraversalAnalysisRequest, LatentTraversalAnalysisResponse, df_describe_index, LatentTraversalAnalysisOutput
from core.repository import InferenceRepository
from core.service import LatentTraversalService, ScalerService


class LatentTraversalAnalysisUseCase:
    def __init__(self, traversal_service: LatentTraversalService,
                 scaler_service: ScalerService,
                 inference_repository: InferenceRepository):

        self.traversal_service = traversal_service
        self.scaler_service = scaler_service
        self.repository = inference_repository
        self.sweep_col_name = 'sweep'

    def run(self, request: LatentTraversalAnalysisRequest)->LatentTraversalAnalysisResponse:
        results: list[LatentTraversalAnalysisOutput] = []
        for input_config in request.configurations:
            result = self.traversal_service.execute(input_config)
            sweep = self._dataframe_output(result.sweeps, result.recon)
            stats = sweep.describe().T
            stats_filtered = input_config.filter.do_filter(stats)
            sweep_r = self._rescale_result(sweep)
            stats_r = self._rescale_result(stats.T)
            stats_filtered_r = self._rescale_result(stats_filtered.T)
            results.append(LatentTraversalAnalysisOutput(
                dimension=input_config.dimension,
                degree_of_freedom=result.degree_of_freedom,
                sweeps=result.sweeps,
                raw_sweep=sweep,
                raw_stats=stats,
                raw_stats_filtered=stats_filtered,
                sweep_rescaled=sweep_r,
                stats_rescaled=stats_r,
                stats_filtered_rescaled=stats_filtered_r,
            ))
        return LatentTraversalAnalysisResponse(results)

    def _dataframe_output(self, sweep, recon):
        df = pd.DataFrame(recon, columns=self.repository.get_feature_columns())
        df.insert(0, self.sweep_col_name, sweep)
        return df

    def _rescale_result(self, result: pd.DataFrame) -> pd.DataFrame:
        result = result.copy()
        is_stats = result.index.isin(df_describe_index).any()
        has_snr = 'snr' in result.index
        has_sweep = self.sweep_col_name in result.columns
        sweep = result[[self.sweep_col_name]] if has_sweep else pd.DataFrame(index=result.index)
        features = result.drop(columns=[self.sweep_col_name]) if has_sweep else result

        if features.empty:
            return result.T if is_stats else result

        if is_stats:
            rescaled_df = self.scaler_service.inverse_transform_df_stats(features)
            output_df = rescaled_df.T
            if has_snr:
                output_df['snr'] = output_df['mean'] / output_df['std'].replace(0, np.nan)
        else:
            rescaled_df = self.scaler_service.inverse_transform_df(features)
            output_df = pd.concat([sweep, rescaled_df], axis=1)

        return output_df
