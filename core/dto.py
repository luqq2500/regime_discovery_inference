from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

from core.domain import LatentTraversalStatsFilter


@dataclass
class LatentTraversalInput:
    dimension: int
    sigma_range: tuple[float, float]
    snr_threshold: float=0.1
    top_n: int=5
    def __post_init__(self):
        self.filter = LatentTraversalStatsFilter(
            snr_threshold=self.snr_threshold,
            top_n=self.top_n
        )

@dataclass
class LatentTraversalOutput:
    dimension: int
    degree_of_freedom: float
    sweeps: list[float]
    recon: np.ndarray

@dataclass
class LatentTraversalAnalysisRequest:
    configurations: list[LatentTraversalInput]

@dataclass
class FeatureMetrics:
    asset: Optional[str]
    horizon: Optional[str]
    mean: Optional[float]
    std: Optional[float]
    snr: Optional[float]
    mean_latent: Optional[float]
    std_latent: Optional[float]
    snr_latent: Optional[float]

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


