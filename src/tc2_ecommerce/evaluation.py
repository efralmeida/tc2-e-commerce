"""Ranking metrics and evaluator helpers for recommendation models."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
  if not relevant:
    return 0.0
  rec_k = recommended[:k]
  hits = len(set(rec_k) & relevant)
  return hits / len(relevant)


def hitrate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
  rec_k = set(recommended[:k])
  return 1.0 if len(rec_k & relevant) > 0 else 0.0


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
  dcg = 0.0
  for i, item in enumerate(recommended[:k]):
    if item in relevant:
      dcg += 1.0 / math.log2(i + 2)
  ideal_hits = min(len(relevant), k)
  if ideal_hits == 0:
    return 0.0
  idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
  return dcg / idcg


def map_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
  if not relevant:
    return 0.0
  hits = 0
  precision_sum = 0.0
  for i, item in enumerate(recommended[:k], start=1):
    if item in relevant:
      hits += 1
      precision_sum += hits / i
  return precision_sum / min(len(relevant), k)


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
  if k <= 0:
    return 0.0
  rec_k = recommended[:k]
  if not rec_k:
    return 0.0
  hits = len(set(rec_k) & relevant)
  return hits / min(len(rec_k), k)


def mrr_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
  for i, item in enumerate(recommended[:k], start=1):
    if item in relevant:
      return 1.0 / i
  return 0.0


class Evaluator:
  """Calculates ranking metrics for recommendation outputs."""

  def __init__(self, k: int = 10) -> None:
    self.k = k

  def evaluate(
    self,
    user_recs: dict[int, list[int]],
    truth: dict[int, set[int]],
  ) -> dict[str, float]:
    users = [u for u in truth if u in user_recs]
    if not users:
      return {
        "recall@k": 0.0,
        "hitrate@k": 0.0,
        "ndcg@k": 0.0,
        "map@k": 0.0,
        "precision@k": 0.0,
        "mrr@k": 0.0,
        "coverage@k": 0.0,
      }

    recalls = []
    hits = []
    ndcgs = []
    maps = []
    precisions = []
    mrrs = []
    recommended_items = []

    for user in users:
      recs = user_recs[user]
      rel = truth[user]
      recalls.append(recall_at_k(recs, rel, self.k))
      hits.append(hitrate_at_k(recs, rel, self.k))
      ndcgs.append(ndcg_at_k(recs, rel, self.k))
      maps.append(map_at_k(recs, rel, self.k))
      precisions.append(precision_at_k(recs, rel, self.k))
      mrrs.append(mrr_at_k(recs, rel, self.k))
      recommended_items.extend(recs[: self.k])

    truth_items = set().union(*truth.values()) if truth else set()
    unique_recommended = set(recommended_items)
    coverage = len(unique_recommended) / max(1, len(truth_items))

    return {
      "recall@k": float(mean(recalls)),
      "hitrate@k": float(mean(hits)),
      "ndcg@k": float(mean(ndcgs)),
      "map@k": float(mean(maps)),
      "precision@k": float(mean(precisions)),
      "mrr@k": float(mean(mrrs)),
      "coverage@k": float(coverage),
    }

  @staticmethod
  def sanitize_metric_name(name: str) -> str:
    return name.replace("@", "_at_")

  @staticmethod
  def recommendation_histogram(
    user_recs: dict[int, list[int]],
    k: int = 10,
  ) -> dict[int, int]:
    flat = []
    for recs in user_recs.values():
      flat.extend(recs[:k])
    return dict(Counter(flat))
