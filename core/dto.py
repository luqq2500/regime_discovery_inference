import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd


df_describe_index = pd.DataFrame({"sweep": [1]}).describe().index.tolist()

@dataclass
class LatentTraversalStatsFilter:
    std_threshold: Optional[float] = None
    snr_threshold: Optional[float] = None
    top_n: Optional[int] = None
    filter_columns: Optional[list[str]] = field(default_factory=lambda: ['mean', 'std'])
    sort_column: Optional[str] = 'snr'

    def __post_init__(self):
        self._validate()

    def _validate(self):
        for filter_col in self.filter_columns:
            if filter_col not in df_describe_index:
                raise ValueError(f'Unknown filter column {filter_col}')

    def _filter_columns(self, df):
        df = df.copy()
        return df[list(set(self.filter_columns))]

    def _filter_snr(self, df):
        df = df.copy()
        df['snr'] = df['mean'].abs() / df['std'].replace(0, np.nan)
        return df[df['snr'] >= self.snr_threshold]

    def _filter_std(self, df):
        df = df.copy()
        return df[df['std'] <= self.std_threshold]

    def _sort_column(self, df):
        df = df.copy()
        return df.sort_values(by=[self.sort_column], ascending=False)

    def _get_top(self, df):
        df = df.copy()
        return df.head(self.top_n)

    def do_filter(self, df: pd.DataFrame):
        df = df.copy()
        if self.filter_columns is not None:
            df = self._filter_columns(df)
        if self.std_threshold is not None:
            df = self._filter_std(df)
        if self.snr_threshold is not None:
            df = self._filter_snr(df)
        if self.sort_column is not None:
            df = self._sort_column(df)
        if self.top_n is not None:
            df = self._get_top(df)
        return df

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
    raw_sweep: pd.DataFrame
    raw_stats: pd.DataFrame
    raw_stats_filtered: pd.DataFrame
    sweep_rescaled: pd.DataFrame
    stats_rescaled: pd.DataFrame
    stats_filtered_rescaled: pd.DataFrame

    def __post_init__(self):
        features = [feature.split("-", 1) for feature in self.stats_filtered_rescaled.index.tolist()]
        assets = [feature[0] if len(feature) > 0 else None for feature in features]
        horizons = [feature[1] if len(feature) > 1 else None for feature in features]

        means = self.stats_filtered_rescaled["mean"].values
        stds = self.stats_filtered_rescaled["std"].values
        snrs = self.stats_filtered_rescaled["snr"].values

        raw_means = self.raw_stats_filtered["mean"].values
        raw_stds = self.raw_stats_filtered["std"].values
        raw_snrs = self.raw_stats_filtered["snr"].values

        '''  
        if 'snr' not in self.stats_filtered_rescaled.columns:
            snrs = np.where(stds != 0, means/stds, np.nan)
        else:
            snrs = self.stats_filtered_rescaled["snr"].values
        '''

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


    def _prepare_feature_metrics_payload(self):
        return [asdict(feature_metric) for feature_metric in self.feature_metrics]

    def _prepare_rescaled_sweep_payload(self):
        return json.loads(self.sweep_rescaled.replace({np.nan: None}).to_json(orient='records', double_precision=3))

    def get_id(self):
        return self.dimension

    def get_payload(self):
        return {
            'dimension': self.dimension,
            'degree of freedom': self.degree_of_freedom,
            'sweeps': self.sweeps,
            #'raw_stats_filtered': self.raw_stats_filtered,
            #'stats_filtered_rescaled': self.stats_filtered_rescaled,
            'feature_metrics': self._prepare_feature_metrics_payload(),
        }

@dataclass
class LatentTraversalAnalysisResponse:
    results: list[LatentTraversalAnalysisOutput]
    def get_payload(self):
        return {output.dimension: output.get_payload() for output in self.results}
