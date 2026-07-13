from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

df_describe_index = pd.DataFrame({"sweep": [1]}).describe().index.tolist()

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
class FeatureMetrics:
    asset: Optional[str]
    horizon: Optional[str]
    mean: Optional[float]
    std: Optional[float]
    snr: Optional[float]
    mean_latent: Optional[float]
    std_latent: Optional[float]
    snr_latent: Optional[float]