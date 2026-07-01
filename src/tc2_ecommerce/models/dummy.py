"""Dummy recommender based on global weighted popularity."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from tc2_ecommerce.models.base import BaseRecommender


class DummyBaseline(BaseRecommender):
    """Recommend the same globally popular items for all users."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self.max_candidates: int = int(self.config.get("max_candidates", 2000))
        self.popular_items: list[int] = []
        self.train_user_items: dict[int, set[int]] = {}

    def fit(self, train_data: pd.DataFrame) -> DummyBaseline:
        """Train popularity ranking from interaction data."""
        self._validate_interactions(train_data)

        self.train_user_items = self.build_user_items(train_data)
        ranking_source = train_data.copy()
        if "weight" not in ranking_source.columns:
            ranking_source["weight"] = 1.0

        self.popular_items = (
            ranking_source.groupby("itemid")["weight"]
            .sum()
            .sort_values(ascending=False)
            .index.astype(int)
            .tolist()[: self.max_candidates]
        )
        self.trained = True
        return self

    def predict(self, user_ids: Sequence[int], k: int = 10) -> dict[int, list[int]]:
        """Predict top-k items per user excluding seen items."""
        self._ensure_trained()

        top_k = int(k)
        predictions: dict[int, list[int]] = {}
        for user in user_ids:
            user_id = int(user)
            seen_items = self.train_user_items.get(user_id, set())
            recs = [item for item in self.popular_items if item not in seen_items][:top_k]
            predictions[user_id] = recs
        return predictions
