from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import torch
from sklearn.base import TransformerMixin


@dataclass
class InferenceParams:
  filter_columns: list[str]
  sigma: float
  std_t: float
  snr_t: float
  sort: tuple[str, int]
  filter_result: list[str]

  def _std_below(self, df, t):
      return df[df['std'] <= t]

  def _snr_above(self, df, t):
      df_copy = df.copy()
      df_copy['snr'] = df_copy['mean'].abs() / df_copy['std'].replace(0, np.nan)
      return df_copy[df_copy['snr'] >= t]

  def _sort(self, df):
      df = df.copy()
      if self.sort is None:
        return df
      column, top_n = self.sort
      if column is not None:
        df = df.sort_values(by=column, ascending=False)
      if top_n is not None:
        df = df.head(top_n)
      return df

  def _filter_columns(self, df):
    df = df.copy()
    if self.filter_columns is None:
      return df
    return df[self.filter_columns]

  def do_filter(self, df):
    df = df.copy()
    df = self._filter_columns(df)
    for filter_func in self.filter_funcs:
      df = filter_func(df)
    df = self._sort(df)
    return df

  def __post_init__(self):
    self.filter_funcs = [
        lambda df: self._std_below(df, self.std_t),
        lambda df: self._snr_above(df, self.snr_t)
    ]

df_stats_index = ['mean', 'std', 'min', 'max']

@dataclass
class LatentTraversalStatsFilter:
    filter_columns: list[str]
    std_threshold: float
    snr_threshold: float
    sort_column: tuple[str, int]

    def __post_init__(self):
        self._validate()

    def _validate(self):
        for filter_col in self.filter_columns:
            if filter_col not in df_stats_index:
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
        sort_col, _ = self.sort_column
        return df.sort_values(by=[sort_col], ascending=False)

    def _get_top(self, df):
        df = df.copy()
        _, top = self.sort_column
        return df.head(top)

    def do_filter(self, df: pd.DataFrame):
        df = df.copy()
        if self.filter_columns is not None:
            df = self._filter_columns(df)
        if self.std_threshold is not None:
            df = self._filter_snr(df)
        if self.snr_threshold is not None:
            df = self._filter_snr(df)
        sort_col, top_n = self.sort_column
        if sort_col is not None:
            df = self._sort_column(df)
        if top_n is not None:
            df = self._get_top(df)
        return df



@dataclass
class LatentTraversalConfig:
    model: torch.nn.Module
    col_names: list[str]
    scaler: TransformerMixin

@dataclass
class LatentTraversalRequest:
    sigma: float
    filter: LatentTraversalStatsFilter

@dataclass
class LatentTraversalOutput:
    dimension: int
    sweep: pd.DataFrame
    stats: pd.DataFrame
    stats_filtered: pd.DataFrame
    sweep_rescaled: pd.DataFrame
    stats_rescaled: pd.DataFrame
    stats_filtered_rescaled: pd.DataFrame

@dataclass
class LatentTraversalResponse:
    response: list[LatentTraversalOutput]