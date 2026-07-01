"""Preprocessing strategies for recommender input data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BasePreprocessor(ABC):
  """Base strategy for preprocessors."""

  @abstractmethod
  def fit(self, data: pd.DataFrame) -> BasePreprocessor:
    """Learn parameters from data."""

  @abstractmethod
  def transform(self, data: pd.DataFrame) -> pd.DataFrame:
    """Apply transformation to data."""

  def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
    self.fit(data)
    return self.transform(data)


class StandardPreprocessor(BasePreprocessor):
  def fit(self, data: pd.DataFrame) -> StandardPreprocessor:
    return self

  def transform(self, data: pd.DataFrame) -> pd.DataFrame:
    return data.copy()


class NormalizationPreprocessor(BasePreprocessor):
  """Min-max normalize numeric columns."""

  def __init__(self) -> None:
    self.min_values: dict[str, float] = {}
    self.max_values: dict[str, float] = {}
    self.columns: list[str] = []

  def fit(self, data: pd.DataFrame) -> NormalizationPreprocessor:
    self.columns = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
    for col in self.columns:
      self.min_values[col] = float(data[col].min())
      self.max_values[col] = float(data[col].max())
    return self

  def transform(self, data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    for col in self.columns:
      min_v = self.min_values[col]
      max_v = self.max_values[col]
      if max_v == min_v:
        frame[col] = 0.0
      else:
        frame[col] = (frame[col] - min_v) / (max_v - min_v)
    return frame


class LogScalingPreprocessor(BasePreprocessor):
  """Apply log1p scaling to selected numeric columns."""

  def __init__(self, columns: list[str] | None = None) -> None:
    self.columns = columns or []

  def fit(self, data: pd.DataFrame) -> LogScalingPreprocessor:
    if not self.columns:
      self.columns = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
    return self

  def transform(self, data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    for col in self.columns:
      frame[col] = np.log1p(np.clip(frame[col], a_min=0, a_max=None))
    return frame


class InteractionWeightingPreprocessor(BasePreprocessor):
  """Map event types to interaction weights used in baseline notebook."""

  def __init__(self, event_weights: dict[str, float] | None = None) -> None:
    self.event_weights = event_weights or {
      "view": 1.0,
      "addtocart": 3.0,
      "transaction": 5.0,
    }

  def fit(self, data: pd.DataFrame) -> InteractionWeightingPreprocessor:
    return self

  def transform(self, data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    if "event" not in frame.columns:
      raise ValueError("Coluna 'event' obrigatória para InteractionWeightingPreprocessor")
    frame = frame[frame["event"].isin(self.event_weights.keys())].copy()
    frame["weight"] = frame["event"].map(self.event_weights).astype(float)
    return frame


class PreprocessorFactory:
  """Factory for preprocessor strategies."""

  _registry: dict[str, type[BasePreprocessor]] = {
    "standard": StandardPreprocessor,
    "normalization": NormalizationPreprocessor,
    "log_scaling": LogScalingPreprocessor,
    "interaction_weighting": InteractionWeightingPreprocessor,
  }

  @classmethod
  def create(cls, name: str, **kwargs: Any) -> BasePreprocessor:
    normalized = name.strip().lower()
    if normalized not in cls._registry:
      available = ", ".join(sorted(cls._registry))
      raise ValueError(f"Preprocessador desconhecido '{name}'. Disponíveis: {available}")
    return cls._registry[normalized](**kwargs)
