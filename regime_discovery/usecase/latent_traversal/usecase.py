import pandas as pd
import torch

from dto import LatentTraversalRequest, LatentTraversalResponse, LatentTraversalConfig, LatentTraversalOutput


class LatentTraversalUseCase:
    def __init__(self, config: LatentTraversalConfig):
        self.model = config.model
        self.column_names = config.col_names
        self.scaler = config.scaler

    def execute(self, request: LatentTraversalRequest) -> LatentTraversalResponse:
        with torch.no_grad():
            outputs = self.model(request.sigma)
        responses: list[LatentTraversalOutput] = []
        for dim, result in outputs.items():
            sweep_values, recon = result
            sweep = self._df_output(sweep_values, recon)
            stats = sweep.describe().T
            stats_filtered = request.filter.do_filter(stats)
            sweep_r = self._rescale_output(sweep)
            stats_r = sweep_r.describe().T
            stats_filtered_r = request.filter.do_filter(stats_r)
            responses.append(LatentTraversalOutput(dim, sweep, stats, stats_filtered, sweep_r, stats_r, stats_filtered_r))
        return LatentTraversalResponse(responses)

    def _df_output(self, sweep, recon, sweep_column='sweep'):
        df = pd.DataFrame(recon, columns=self.column_names)
        df.insert(0, sweep_column, sweep)
        return df

    def _rescale_output(self, output: pd.DataFrame) -> pd.DataFrame:
        output = output.copy()
        sweep = output.iloc[:, 0]
        features = output.iloc[:, 1:]
        rescaled = self.scaler.inverse_transform(features.values)
        rescaled_output = pd.DataFrame(rescaled, columns=features.columns, index=features.index)
        rescaled_output = pd.concat([sweep, rescaled_output], axis=1)
        return rescaled_output
