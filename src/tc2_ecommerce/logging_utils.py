"""Logging utilities for training, evaluation and API services."""

from __future__ import annotations

import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from tc2_ecommerce.config import settings


def setup_logging(log_dir: str | Path = "logs") -> logging.Logger:
  """Configure root logger with console + rotating file handlers."""
  target_dir = Path(log_dir)
  target_dir.mkdir(parents=True, exist_ok=True)

  logger = logging.getLogger()
  logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

  if logger.handlers:
    return logger

  formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
  )

  console_handler = logging.StreamHandler()
  console_handler.setFormatter(formatter)

  file_handler = RotatingFileHandler(
    target_dir / "tc2-ecommerce.log",
    maxBytes=2_000_000,
    backupCount=3,
    encoding="utf-8",
  )
  file_handler.setFormatter(formatter)

  logger.addHandler(console_handler)
  logger.addHandler(file_handler)
  return logger


def get_logger(name: str) -> logging.Logger:
  """Return named logger after global setup."""
  setup_logging()
  return logging.getLogger(name)


def log_experiment(
  logger: logging.Logger,
  model_name: str,
  params: dict[str, Any],
  metrics: dict[str, float],
) -> None:
  payload = {
    "event": "experiment",
    "model": model_name,
    "params": params,
    "metrics": metrics,
  }
  logger.info(json.dumps(payload, ensure_ascii=True))


def log_performance(
  logger: logging.Logger,
  stage: str,
  duration_seconds: float,
  memory_used_mb: float | None = None,
) -> None:
  payload = {
    "event": "performance",
    "stage": stage,
    "duration_seconds": round(duration_seconds, 4),
    "memory_used_mb": memory_used_mb,
    "timestamp": time.time(),
  }
  logger.info(json.dumps(payload, ensure_ascii=True))
