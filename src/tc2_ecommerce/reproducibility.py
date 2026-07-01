"""Reproducibility helpers for deterministic experiments."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
  import torch

  _TORCH_AVAILABLE = True
except ImportError:
  _TORCH_AVAILABLE = False


def set_seed(seed: int = 42) -> None:
  """Set random seeds across common libs."""
  random.seed(seed)
  np.random.seed(seed)

  if _TORCH_AVAILABLE:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

  os.environ["PYTHONHASHSEED"] = str(seed)


def get_git_commit_hash() -> str | None:
  """Return current git commit hash when available."""
  try:
    result = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      capture_output=True,
      text=True,
      check=True,
    )
    return result.stdout.strip()
  except Exception:
    return None


def get_system_info() -> dict[str, Any]:
  """Collect system and runtime metadata for experiment traceability."""
  info: dict[str, Any] = {
    "python_version": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "git_commit": get_git_commit_hash(),
    "numpy_version": np.__version__,
  }

  if _TORCH_AVAILABLE:
    info["torch_version"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
      info["cuda_device_count"] = torch.cuda.device_count()
      info["cuda_device_name"] = torch.cuda.get_device_name(0)
  else:
    info["torch_version"] = None
    info["cuda_available"] = False
    info["cuda_version"] = None

  return info


def write_system_info(path: str | Path) -> Path:
  """Persist system metadata JSON to disk."""
  target = Path(path)
  target.parent.mkdir(parents=True, exist_ok=True)
  with target.open("w", encoding="utf-8") as f:
    json.dump(get_system_info(), f, indent=2)
  return target


def log_reproducibility(mlflow_module: Any, seed: int = 42) -> None:
  """Log reproducibility metadata to mlflow-like module."""
  mlflow_module.log_param("seed", seed)
  info = get_system_info()
  for key, value in info.items():
    mlflow_module.log_param(f"system_{key}", str(value))
