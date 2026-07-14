from dataclasses import dataclass

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
    regime: int
    degree_of_freedom: float
    traversal_sweeps: list[float]
    result_scaled: pd.DataFrame
    result_raw: pd.DataFrame
    asset_stats_scaled: pd.DataFrame
    asset_stats_raw: pd.DataFrame
    filtered_asset_stats_scaled: pd.DataFrame
    filtered_asset_stats_raw: pd.DataFrame

@dataclass
class LatentTraversalAnalysisResponse:
    outputs: list[LatentTraversalAnalysisOutput]


