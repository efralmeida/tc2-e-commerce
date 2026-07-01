"""Neural recommender using user/item embeddings with PyTorch."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from tc2_ecommerce.models.base import BaseRecommender


class _NeuralCF(nn.Module):
  """Simple MLP over concatenated user/item embeddings."""

  def __init__(self, num_users: int, num_items: int, emb_dim: int, dropout: float) -> None:
    super().__init__()
    self.user_embedding = nn.Embedding(num_users, emb_dim)
    self.item_embedding = nn.Embedding(num_items, emb_dim)
    self.mlp = nn.Sequential(
      nn.Linear(emb_dim * 2, emb_dim),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(emb_dim, emb_dim // 2 if emb_dim > 1 else 1),
      nn.ReLU(),
      nn.Linear(emb_dim // 2 if emb_dim > 1 else 1, 1),
    )

  def forward(self, user_idx: Tensor, item_idx: Tensor) -> Tensor:
    user_vec = self.user_embedding(user_idx)
    item_vec = self.item_embedding(item_idx)
    features = torch.cat([user_vec, item_vec], dim=1)
    return self.mlp(features).squeeze(1)


class NeuralRecommender(BaseRecommender):
  """Embedding-based recommender with implicit feedback training."""

  def __init__(self, config: dict[str, Any] | None = None) -> None:
    super().__init__(config=config)
    self.embedding_dim: int = int(self.config.get("embedding_dim", 128))
    self.dropout: float = float(self.config.get("dropout", 0.2))
    self.learning_rate: float = float(self.config.get("learning_rate", 1e-3))
    self.batch_size: int = int(self.config.get("batch_size", 256))
    self.epochs: int = int(self.config.get("epochs", 5))
    self.negatives_per_positive: int = int(self.config.get("negatives_per_positive", 2))
    self.seed: int = int(self.config.get("seed", 42))

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    self.device_name: str = str(self.config.get("device", default_device))
    self.model: _NeuralCF | None = None

    self.user_to_idx: dict[int, int] = {}
    self.idx_to_user: dict[int, int] = {}
    self.item_to_idx: dict[int, int] = {}
    self.idx_to_item: dict[int, int] = {}
    self.train_user_items: dict[int, set[int]] = {}
    self.all_items: list[int] = []

  def fit(self, train_data: pd.DataFrame) -> NeuralRecommender:
    """Train neural recommender using implicit positives and sampled negatives."""
    self._validate_interactions(train_data)
    self._set_seeds()

    self.train_user_items = self.build_user_items(train_data)
    self.all_items = sorted({int(item) for item in train_data["itemid"].unique()})
    all_users = sorted({int(user) for user in train_data["visitorid"].unique()})

    self.user_to_idx = {user_id: idx for idx, user_id in enumerate(all_users)}
    self.idx_to_user = {idx: user_id for user_id, idx in self.user_to_idx.items()}
    self.item_to_idx = {item_id: idx for idx, item_id in enumerate(self.all_items)}
    self.idx_to_item = {idx: item_id for item_id, idx in self.item_to_idx.items()}

    dataset = self._build_training_dataset()
    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    self.model = _NeuralCF(
      num_users=len(self.user_to_idx),
      num_items=len(self.item_to_idx),
      emb_dim=self.embedding_dim,
      dropout=self.dropout,
    ).to(self._device())

    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    self.model.train()

    for _ in range(self.epochs):
      for user_tensor, item_tensor, label_tensor in dataloader:
        user_tensor = user_tensor.to(self._device())
        item_tensor = item_tensor.to(self._device())
        label_tensor = label_tensor.to(self._device())

        optimizer.zero_grad()
        logits = self.model(user_tensor, item_tensor)
        loss = criterion(logits, label_tensor)
        loss.backward()
        optimizer.step()

    self.trained = True
    return self

  def predict(self, user_ids: Sequence[int], k: int = 10) -> dict[int, list[int]]:
    """Predict top-k items by neural score for each user."""
    self._ensure_trained()
    if self.model is None:
      raise RuntimeError("Modelo neural não inicializado.")

    self.model.eval()
    top_k = int(k)
    predictions: dict[int, list[int]] = {}

    for user in user_ids:
      user_id = int(user)
      if user_id not in self.user_to_idx:
        predictions[user_id] = []
        continue

      seen_items = self.train_user_items.get(user_id, set())
      candidates = [item for item in self.all_items if item not in seen_items]
      if not candidates:
        predictions[user_id] = []
        continue

      user_idx_value = self.user_to_idx[user_id]
      user_idx_tensor = torch.full(
        (len(candidates),),
        fill_value=user_idx_value,
        dtype=torch.long,
        device=self._device(),
      )
      item_idx_tensor = torch.tensor(
        [self.item_to_idx[item] for item in candidates],
        dtype=torch.long,
        device=self._device(),
      )

      with torch.no_grad():
        scores = torch.sigmoid(self.model(user_idx_tensor, item_idx_tensor))

      scores_np = scores.detach().cpu().numpy()
      top_idx = np.argsort(scores_np)[::-1][:top_k]
      predictions[user_id] = [int(candidates[i]) for i in top_idx]

    return predictions

  def save(self, path: str | Path) -> None:
    """Persist neural weights plus mappings."""
    self._ensure_trained()
    if self.model is None:
      raise RuntimeError("Modelo neural não inicializado.")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
      "state_dict": self.model.state_dict(),
      "config": self.config,
      "user_to_idx": self.user_to_idx,
      "idx_to_user": self.idx_to_user,
      "item_to_idx": self.item_to_idx,
      "idx_to_item": self.idx_to_item,
      "train_user_items": self.train_user_items,
      "all_items": self.all_items,
      "trained": self.trained,
    }
    torch.save(payload, target)

  @classmethod
  def load(cls, path: str | Path) -> NeuralRecommender:
    """Load neural model state and metadata."""
    payload = torch.load(Path(path), map_location="cpu")
    model = cls(config=dict(payload.get("config", {})))
    model.user_to_idx = dict(payload["user_to_idx"])
    model.idx_to_user = dict(payload["idx_to_user"])
    model.item_to_idx = dict(payload["item_to_idx"])
    model.idx_to_item = dict(payload["idx_to_item"])
    model.train_user_items = dict(payload["train_user_items"])
    model.all_items = list(payload["all_items"])

    model.model = _NeuralCF(
      num_users=len(model.user_to_idx),
      num_items=len(model.item_to_idx),
      emb_dim=model.embedding_dim,
      dropout=model.dropout,
    )
    model.model.load_state_dict(payload["state_dict"])
    model.model.to(model._device())
    model.trained = bool(payload.get("trained", True))
    return model

  def _build_training_dataset(self) -> TensorDataset:
    samples_user: list[int] = []
    samples_item: list[int] = []
    samples_label: list[float] = []

    for user_id, positive_items in self.train_user_items.items():
      user_idx = self.user_to_idx[user_id]
      positives = list(positive_items)
      positive_set = set(positives)

      for item_id in positives:
        samples_user.append(user_idx)
        samples_item.append(self.item_to_idx[item_id])
        samples_label.append(1.0)

        neg_added = 0
        while neg_added < self.negatives_per_positive:
          candidate_item = random.choice(self.all_items)
          if candidate_item in positive_set:
            continue
          samples_user.append(user_idx)
          samples_item.append(self.item_to_idx[candidate_item])
          samples_label.append(0.0)
          neg_added += 1

    users = torch.tensor(samples_user, dtype=torch.long)
    items = torch.tensor(samples_item, dtype=torch.long)
    labels = torch.tensor(samples_label, dtype=torch.float32)
    return TensorDataset(users, items, labels)

  def _device(self) -> torch.device:
    return torch.device(self.device_name)

  def _set_seeds(self) -> None:
    random.seed(self.seed)
    np.random.seed(self.seed)
    torch.manual_seed(self.seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(self.seed)
