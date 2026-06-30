import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.base import TransformerMixin

describe_index = pd.DataFrame({"sweep": [1]}).describe().index.tolist()

@dataclass
class LatentTraversalStatsFilter:
    std_threshold: Optional[float]
    snr_threshold: Optional[float]
    top_n: Optional[int]
    filter_columns: Optional[list[str]] = field(default_factory=lambda: ['mean', 'std'])
    sort_column: Optional[str] = 'snr'

    def __post_init__(self):
        self._validate()

    def _validate(self):
        for filter_col in self.filter_columns:
            if filter_col not in describe_index:
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
class LatentTraversalConfig:
    model: torch.nn.Module
    col_names: list[str]
    scaler: TransformerMixin

@dataclass
class LatentTraversalRequest:
    sigma_range: tuple[float, float]
    filter: LatentTraversalStatsFilter

@dataclass
class LatentTraversalOutput:
    dimension: int
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
        if "snr" not in self.stats_filtered_rescaled.columns:
            snrs = np.where(stds != 0, means/stds, np.nan)
        else:
            snrs = self.stats_filtered_rescaled["snr"].values

        self.feature_metrics: list[OutputFeatureMetric] = [
            OutputFeatureMetric(asset, horizon, round(float(mean),2), round(float(std),2), round(float(snr),2))
            for asset, horizon, mean, std, snr in zip(assets, horizons, means, stds, snrs)
        ]


    def _prepare_feature_metrics_payload(self):
        return [asdict(feature_metric) for feature_metric in self.feature_metrics]

    def _prepare_raw_sweep_payload(self):
        return json.loads(self.raw_sweep.replace({np.nan: None}).to_json(orient='records', double_precision=3))

    def get_payload(self):
        return {
            'dimension': self.dimension,
            'feature_metrics': self._prepare_feature_metrics_payload(),
            'sweep': self._prepare_raw_sweep_payload()
        }


@dataclass
class OutputFeatureMetric:
    asset: Optional[str]
    horizon: Optional[str]
    mean: Optional[float]
    std: Optional[float]
    snr: Optional[float]

@dataclass
class RegimeProfile:
    id: int
    top_assets: Optional[list[OutputFeatureMetric]]

@dataclass
class LatentTraversalResponse:
    outputs: list[LatentTraversalOutput]


