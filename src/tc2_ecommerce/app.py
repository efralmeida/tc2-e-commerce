"""High-level app orchestration for training and serving recommenders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tc2_ecommerce.config import settings
from tc2_ecommerce.factories.model_factory import ModelFactory
from tc2_ecommerce.logging_utils import get_logger, setup_logging
from tc2_ecommerce.train import grid_search, train_model


class App:
  """Application service facade used by CLI and API layers."""

  def __init__(self) -> None:
    setup_logging()
    self.logger = get_logger(__name__)
    self._models_cache: dict[str, Any] = {}

  def load_splits(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/valid/test datasets produced by prepare_data pipeline."""
    processed = Path(settings.data_processed_path)
    train_df = pd.read_csv(processed / "train.csv")
    valid_df = pd.read_csv(processed / "valid.csv")
    test_df = pd.read_csv(processed / "test.csv")
    return train_df, valid_df, test_df

  def train(self, model_name: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Train a model on train split and validate on valid split."""
    train_df, valid_df, _ = self.load_splits()
    result = train_model(
      model_name=model_name,
      train_data=train_df,
      valid_data=valid_df,
      config=config,
      k=settings.top_k,
      run_name=f"app_train_{model_name}",
    )
    self._models_cache[model_name] = result["model"]
    return result

  def tune(
    self,
    model_name: str,
    search_space: dict[str, list[Any]],
    optimize_metric: str = "ndcg@k",
  ) -> dict[str, Any]:
    """Run grid search for a selected model."""
    train_df, valid_df, _ = self.load_splits()
    return grid_search(
      model_name=model_name,
      train_data=train_df,
      valid_data=valid_df,
      search_space=search_space,
      optimize_metric=optimize_metric,
      k=settings.top_k,
    )

  def evaluate(self, model_name: str) -> dict[str, float]:
    """Evaluate trained model using test split."""
    _, _, test_df = self.load_splits()
    model = self.get_model(model_name)
    return model.evaluate(test_df, k=settings.top_k)

  def predict(self, model_name: str, user_ids: list[int], k: int | None = None) -> dict[int, list[int]]:
    """Return top-k recommendations for provided users."""
    model = self.get_model(model_name)
    return model.predict(user_ids, k=int(k or settings.top_k))

  def get_model(self, model_name: str) -> Any:
    """Fetch model from cache, disk, or instantiate a fresh one."""
    if model_name in self._models_cache:
      return self._models_cache[model_name]

    model_path = Path(settings.models_path) / f"{model_name}.pkl"
    if model_path.exists():
      model = ModelFactory.create(model_name, {})
      model = model.__class__.load(model_path)
      self._models_cache[model_name] = model
      return model

    raise ValueError(
      f"Modelo '{model_name}' não encontrado em cache ou em {model_path}. "
      "Execute train() primeiro."
    )
