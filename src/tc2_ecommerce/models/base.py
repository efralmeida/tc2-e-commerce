"""Base abstractions and shared ranking metrics for recommenders."""

from __future__ import annotations

import math
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import pandas as pd


class BaseRecommender(ABC):
  """Common interface for all recommendation models in the project."""

  def __init__(self, config: dict[str, Any] | None = None) -> None:
    self.config: dict[str, Any] = dict(config or {})
    self.name: str = self.__class__.__name__
    self.version: str = str(self.config.get("version", "0.1.0"))
    self.trained: bool = False

  @abstractmethod
  def fit(self, train_data: pd.DataFrame) -> BaseRecommender:
    """Train the model from interaction data."""

  @abstractmethod
  def predict(self, user_ids: Sequence[int], k: int = 10) -> dict[int, list[int]]:
    """Predict top-k recommendations for a list of users."""

  def evaluate(self, test_data: pd.DataFrame, k: int = 10) -> dict[str, float]:
    """Evaluate model performance with ranking metrics."""
    self._ensure_trained()
    self._validate_interactions(test_data)

    truth = self.build_user_items(test_data)
    users = list(truth.keys())
    predictions = self.predict(users, k=k)
    return self.evaluate_model(predictions, truth, k=k)

  def save(self, path: str | Path) -> None:
    """Serialize the model instance with pickle."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as file_obj:
      pickle.dump(self, file_obj)

  @classmethod
  def load(cls, path: str | Path) -> BaseRecommender:
    """Load a serialized model instance from disk."""
    with Path(path).open("rb") as file_obj:
      model = pickle.load(file_obj)
    if not isinstance(model, cls):
      raise TypeError(f"Modelo carregado não é do tipo esperado: {cls.__name__}")
    return model

  @staticmethod
  def build_user_items(frame: pd.DataFrame) -> dict[int, set[int]]:
    """Build ground-truth map from interaction table."""
    BaseRecommender._validate_interactions(frame)
    grouped = frame.groupby("visitorid")["itemid"].apply(set)
    return {int(user): {int(item) for item in items} for user, items in grouped.items()}

  @staticmethod
  def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
      return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & relevant)
    return hits / len(relevant)

  @staticmethod
  def hitrate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    rec_k = set(recommended[:k])
    return 1.0 if len(rec_k & relevant) > 0 else 0.0

  @staticmethod
  def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    dcg = 0.0
    for index, item in enumerate(recommended[:k]):
      if item in relevant:
        dcg += 1.0 / math.log2(index + 2)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
      return 0.0
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg

  @staticmethod
  def map_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
      return 0.0
    hits = 0
    precision_sum = 0.0
    for index, item in enumerate(recommended[:k], start=1):
      if item in relevant:
        hits += 1
        precision_sum += hits / index
    return precision_sum / min(len(relevant), k)

  @classmethod
  def evaluate_model(
    cls,
    user_recs: dict[int, list[int]],
    truth: dict[int, set[int]],
    k: int = 10,
  ) -> dict[str, float]:
    """Compute average ranking metrics over users."""
    users = [user for user in truth if user in user_recs]
    if not users:
      return {"recall@k": 0.0, "hitrate@k": 0.0, "ndcg@k": 0.0, "map@k": 0.0}

    recalls = []
    hits = []
    ndcgs = []
    maps = []
    for user in users:
      recs = user_recs[user]
      rel = truth[user]
      recalls.append(cls.recall_at_k(recs, rel, k))
      hits.append(cls.hitrate_at_k(recs, rel, k))
      ndcgs.append(cls.ndcg_at_k(recs, rel, k))
      maps.append(cls.map_at_k(recs, rel, k))

    return {
      "recall@k": float(mean(recalls)),
      "hitrate@k": float(mean(hits)),
      "ndcg@k": float(mean(ndcgs)),
      "map@k": float(mean(maps)),
    }

  @staticmethod
  def _validate_interactions(frame: pd.DataFrame) -> None:
    required_columns = {"visitorid", "itemid"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
      missing = ", ".join(sorted(missing_columns))
      raise ValueError(f"DataFrame sem colunas obrigatórias: {missing}")

  def _ensure_trained(self) -> None:
    if not self.trained:
      raise RuntimeError("Modelo ainda não foi treinado. Execute fit() antes.")
