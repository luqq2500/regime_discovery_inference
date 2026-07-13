from dataclasses import dataclass
from typing import Optional

import pandas as pd

from core.domain import LatentTraversalInput, FeatureMetrics


@dataclass
class LatentTraversalAnalysisRequest:
    dimensions: list[int]
    sigma_range: tuple[float, float]
    top_n: int=10
    def __post_init__(self):
        self.inputs = [
            LatentTraversalInput(dimension=dimension, sigma_range=self.sigma_range, top_n=self.top_n)
            for dimension in self.dimensions
        ]

@dataclass
class LatentTraversalAnalysisOutput:
    dimension: int
    degree_of_freedom: float
    sweeps: list[float]
    result_scaled: pd.DataFrame
    stats_scaled: pd.DataFrame
    stats_filtered_scaled: pd.DataFrame
    result_raw: pd.DataFrame
    stats_raw: pd.DataFrame
    stats_filtered_raw: pd.DataFrame
    feature_metrics: list[FeatureMetrics] = None

    def __post_init__(self):
        features = [feature.split("-", 1) for feature in self.stats_filtered_raw.index.tolist()]
        assets = [feature[0] if len(feature) > 0 else None for feature in features]
        horizons = [feature[1] if len(feature) > 1 else None for feature in features]

        means_r = self.stats_filtered_raw["mean"].values
        stds_r = self.stats_filtered_raw["std"].values
        snrs_r = self.stats_filtered_raw["snr"].values

        means_s = self.stats_filtered_scaled["mean"].values
        stds_s = self.stats_filtered_scaled["std"].values
        snrs_s = self.stats_filtered_scaled["snr"].values

        self.feature_metrics = [
            FeatureMetrics(asset, horizon,
                           mean_scaled=round(float(mean_s), 2),
                           std_scaled=round(float(std_s), 2),
                           snr_scaled=round(float(snr_s), 2),
                           mean_raw=round(float(mean_r), 2),
                           std_raw=round(float(std_r), 2),
                           snr_raw=round(float(snr_r), 2),
                           )
            for asset, horizon, mean_s, std_s, snr_s, mean_r, std_r, snr_r in zip(
                assets, horizons, means_s, stds_s, snrs_s, means_r, stds_r, snrs_r)
        ]

@dataclass
class LatentTraversalAnalysisResponse:
    outputs: list[LatentTraversalAnalysisOutput]


