"""
Métricas de avaliação para modelos de recomendação.

Contém funções para calcular recall@k, hitrate@k, nDCG@k, MAP@k, etc.
Centraliza cálculo de métricas usado por baseline_experiments.ipynb

TODO: Mover funções de métrica do notebook para cá:
  - recall_at_k(predictions, targets, k)
  - hitrate_at_k(predictions, targets, k)
  - ndcg_at_k(predictions, targets, k)
  - map_at_k(predictions, targets, k)

TODO: Adicionar outras métricas:
  - precision_at_k
  - mrr_at_k (Mean Reciprocal Rank)
  - coverage
  - diversity (similaridade entre itens recomendados)
  - novelty (ranking de items menos populares)

TODO: Implementar classe Evaluator que:
  - Calcula múltiplas métricas em lote
  - Retorna dict estruturado com resultados
  - Integra com MLflow para logging

TODO: Adicionar visualizações de métricas
TODO: Adicionar comparação entre modelos
"""

# TODO: import numpy as np
# TODO: from typing import Dict, List, Tuple
# TODO: Implementar funções de métrica aqui
