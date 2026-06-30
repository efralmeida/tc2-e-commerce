"""
Modelo Dummy: recomenda items mais populares.

Este é o baseline mais simples - sempre recomenda os mesmos items
para todos os usuários (aqueles com maior número de interações).

TODO: Migrar implementação do notebook/baseline_experiments.ipynb:
  - Adaptar código das células Etapa 5 (Dummy baseline)
  - Herdar de BaseRecommender
  - Implementar fit(), predict(), evaluate()
  - Adicionar parametrização de MAX_ITEMS_BY_POPULARITY

TODO: Implementar fit(train_data):
  - Contar popularidade por item
  - Ordenar items por popularidade decrescente
  - Armazenar top-k items populares

TODO: Implementar predict(user_ids, k):
  - Retornar sempre os mesmos k items populares
  - Independentemente do user_id

TODO: Adicionar logging de hiperparâmetros no MLflow
TODO: Adicionar tempo de execução (profile)
TODO: Adicionar validação de entrada
TODO: Testar com dados de exemplo
"""

# TODO: import pandas as pd
# TODO: from collections import Counter
# TODO: from src.tc2_ecommerce.models.base import BaseRecommender
# TODO: 
# TODO: class DummyBaseline(BaseRecommender):
# TODO:     def __init__(self, config: dict):
# TODO:         self.config = config
# TODO:         self.popular_items = []
# TODO:         self.trained = False
# TODO:     
# TODO:     def fit(self, train_data):
# TODO:         pass
# TODO:     
# TODO:     def predict(self, user_ids, k=10):
# TODO:         pass
