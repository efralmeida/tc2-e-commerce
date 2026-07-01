"""Item-item collaborative filtering via scalable co-visitation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import pandas as pd

from tc2_ecommerce.models.base import BaseRecommender


class ItemItemCF(BaseRecommender):
  """Recommend related items based on co-visitation counts."""

  def __init__(self, config: dict[str, Any] | None = None) -> None:
    super().__init__(config=config)
    self.max_items_by_popularity: int = int(
      self.config.get("max_items_by_popularity", 5000)
    )
    self.max_items_per_user: int = int(self.config.get("max_items_per_user", 30))
    self.top_neighbors: int = int(self.config.get("top_neighbors", 100))

    self.train_user_items: dict[int, set[int]] = {}
    self.neighbors: dict[int, list[tuple[int, float]]] = {}

  def fit(self, train_data: pd.DataFrame) -> ItemItemCF:
    """Build item-to-item neighborhood map with co-visitation."""
    self._validate_interactions(train_data)

    ranking_source = train_data.copy()
    if "weight" not in ranking_source.columns:
      ranking_source["weight"] = 1.0

    popular_items = (
      ranking_source.groupby("itemid")["weight"]
      .sum()
      .sort_values(ascending=False)
      .index.astype(int)
      .tolist()[: self.max_items_by_popularity]
    )
    popular_set = set(popular_items)

    filtered = ranking_source[ranking_source["itemid"].isin(popular_set)].copy()
    self.train_user_items = self.build_user_items(filtered)
    user_sequences = filtered.groupby("visitorid")["itemid"].apply(list).to_dict()

    co_counts: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for items in user_sequences.values():
      unique_items = list(dict.fromkeys(int(item) for item in items))
      unique_items = unique_items[: self.max_items_per_user]
      sequence_size = len(unique_items)
      for i in range(sequence_size):
        item_i = unique_items[i]
        for j in range(i + 1, sequence_size):
          item_j = unique_items[j]
          co_counts[item_i][item_j] += 1.0
          co_counts[item_j][item_i] += 1.0

    neighbors: dict[int, list[tuple[int, float]]] = {}
    for item_i, related in co_counts.items():
      top_related = sorted(related.items(), key=lambda x: x[1], reverse=True)
      neighbors[item_i] = top_related[: self.top_neighbors]

    self.neighbors = neighbors
    self.trained = True
    return self

  def predict(self, user_ids: Sequence[int], k: int = 10) -> dict[int, list[int]]:
    """Score candidates from item neighbors and return top-k."""
    self._ensure_trained()

    top_k = int(k)
    predictions: dict[int, list[int]] = {}
    for user in user_ids:
      user_id = int(user)
      seen = self.train_user_items.get(user_id, set())
      if not seen:
        predictions[user_id] = []
        continue

      scores: dict[int, float] = defaultdict(float)
      for seen_item in seen:
        for candidate, score in self.neighbors.get(seen_item, []):
          if candidate not in seen:
            scores[candidate] += score

      ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
      predictions[user_id] = [item for item, _ in ranked[:top_k]]
    return predictions
