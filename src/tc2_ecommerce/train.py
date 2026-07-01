"""Model training and tuning routines with optional MLflow logging."""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any

import pandas as pd

from tc2_ecommerce.config import settings
from tc2_ecommerce.evaluation import Evaluator
from tc2_ecommerce.factories.model_factory import ModelFactory
from tc2_ecommerce.logging_utils import get_logger, log_experiment, log_performance
from tc2_ecommerce.reproducibility import log_reproducibility, set_seed

try:
  import mlflow

  _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - ambiente sem mlflow
  _MLFLOW_AVAILABLE = False


logger = get_logger(__name__)


def train_model(
  model_name: str,
  train_data: pd.DataFrame,
  valid_data: pd.DataFrame,
  config: dict[str, Any] | None = None,
  *,
  k: int | None = None,
  run_name: str | None = None,
  log_to_mlflow: bool = True,
  save_model: bool = True,
) -> dict[str, Any]:
  """Train model, evaluate on validation split, and optionally log artifacts."""
  local_config = dict(config or {})
  top_k = int(k or settings.top_k)
  seed = int(local_config.get("seed", 42))
  set_seed(seed)

  start = time.time()
  model = ModelFactory.create(model_name, local_config)
  model.fit(train_data)
  metrics = model.evaluate(valid_data, k=top_k)
  elapsed = time.time() - start

  model_path = None
  if save_model:
    target = Path(settings.models_path)
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / f"{model_name}.pkl"
    model.save(model_path)

  if log_to_mlflow and _MLFLOW_AVAILABLE:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name=run_name or f"train_{model_name}"):
      mlflow.log_param("model_name", model_name)
      mlflow.log_param("top_k", top_k)
      for key, value in local_config.items():
        mlflow.log_param(str(key), str(value))
      for metric_name, metric_value in metrics.items():
        mlflow.log_metric(Evaluator.sanitize_metric_name(metric_name), metric_value)
      mlflow.log_metric("runtime_seconds", elapsed)
      log_reproducibility(mlflow, seed=seed)
      if model_path is not None and model_path.exists():
        mlflow.log_artifact(str(model_path))

  log_experiment(logger, model_name=model_name, params=local_config, metrics=metrics)
  log_performance(logger, stage=f"train_{model_name}", duration_seconds=elapsed)

  return {
    "model": model,
    "metrics": metrics,
    "model_path": str(model_path) if model_path is not None else None,
    "runtime_seconds": elapsed,
    "config": local_config,
  }


def grid_search(
  model_name: str,
  train_data: pd.DataFrame,
  valid_data: pd.DataFrame,
  search_space: dict[str, list[Any]],
  *,
  optimize_metric: str = "ndcg@k",
  k: int | None = None,
) -> dict[str, Any]:
  """Run exhaustive grid search and return best run by selected metric."""
  if not search_space:
    raise ValueError("search_space não pode ser vazio")

  keys = list(search_space.keys())
  values_product = itertools.product(*(search_space[key] for key in keys))

  results: list[dict[str, Any]] = []
  best_result: dict[str, Any] | None = None

  for values in values_product:
    params = dict(zip(keys, values, strict=True))
    run_label = "_".join(f"{k}_{v}" for k, v in params.items())
    result = train_model(
      model_name=model_name,
      train_data=train_data,
      valid_data=valid_data,
      config=params,
      k=k,
      run_name=f"grid_{model_name}_{run_label}",
      log_to_mlflow=True,
      save_model=False,
    )
    enriched = {"params": params, **result}
    results.append(enriched)

    metric_value = float(result["metrics"].get(optimize_metric, float("-inf")))
    if best_result is None:
      best_result = enriched
    else:
      best_metric = float(best_result["metrics"].get(optimize_metric, float("-inf")))
      if metric_value > best_metric:
        best_result = enriched

  if best_result is None:
    raise RuntimeError("Nenhum resultado encontrado durante grid_search")

  return {
    "best": best_result,
    "all_results": results,
    "optimize_metric": optimize_metric,
  }
