from dataclasses import dataclass

import pandas as pd

from core.domain import LatentTraversalInput, FeatureMetrics

@dataclass
class LatentTraversalAnalysisRequest:
    inputs: list[LatentTraversalInput]

@dataclass
class LatentTraversalAnalysisOutput:
    dimension: int
    degree_of_freedom: float
    sweeps: list[float]
    df_raw: pd.DataFrame
    stats_raw: pd.DataFrame
    stats_filtered_raw: pd.DataFrame
    df_rescaled: pd.DataFrame
    stats_rescaled: pd.DataFrame
    stats_filtered_rescaled: pd.DataFrame

    def __post_init__(self):
        features = [feature.split("-", 1) for feature in self.stats_filtered_rescaled.index.tolist()]
        assets = [feature[0] if len(feature) > 0 else None for feature in features]
        horizons = [feature[1] if len(feature) > 1 else None for feature in features]

        means = self.stats_filtered_rescaled["mean"].values
        stds = self.stats_filtered_rescaled["std"].values
        snrs = self.stats_filtered_rescaled["snr"].values

        raw_means = self.stats_filtered_raw["mean"].values
        raw_stds = self.stats_filtered_raw["std"].values
        raw_snrs = self.stats_filtered_raw["snr"].values

        self.feature_metrics: list[FeatureMetrics] = [
            FeatureMetrics(asset, horizon,
                           round(float(mean), 2),
                           round(float(std), 2),
                           round(float(snr), 2),
                           round(float(raw_mean), 2),
                           round(float(raw_std), 2),
                           round(float(raw_snr), 2))
            for asset, horizon, mean, std, snr, raw_mean, raw_std, raw_snr in zip(
                assets, horizons, means, stds, snrs, raw_means, raw_stds, raw_snrs)
        ]

@dataclass
class LatentTraversalAnalysisResponse:
    results: list[LatentTraversalAnalysisOutput]


