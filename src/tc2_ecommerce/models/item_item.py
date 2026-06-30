"""
Modelo Item-Item Collaborative Filtering.

Recomenda items similares aos que o usuário interagiu antes.
Usa abordagem escalável com co-visitation (não precisa matriz densa).

TODO: Migrar implementação do notebook/baseline_experiments.ipynb:
  - Adaptar código das células Etapa 6 (Item-item collaborative)
  - Herdar de BaseRecommender
  - Implementar fit(), predict(), evaluate()
  - Manter defaultdict para co-visitation (escalável)

TODO: Implementar fit(train_data):
  - Criar mapa de co-visitation (quais items aparecem juntos)
  - Ordenar neighbors por similaridade
  - Limitar a TOP_NEIGHBORS para performance

TODO: Implementar predict(user_ids, k):
  - Para cada user, buscar items que interagiu
  - Para cada item do user, buscar items similares (neighbors)
  - Retornar top-k recommendations (ordenadas por score)

TODO: Adicionar parametrização:
  - MAX_ITEMS_PER_USER: limitar items por usuário
  - TOP_NEIGHBORS: limitar neighbors por item
  - SIMILARITY_METRIC: outras formas de calcular similaridade

TODO: Adicionar logging de hiperparâmetros no MLflow
TODO: Adicionar tempo de execução e profile de memória
TODO: Adicionar validação de entrada
TODO: Testar escalabilidade com dados grandes

TODO: Possível extensão: usar similaridade baseada em embedding (futuro neural)
"""

# TODO: import pandas as pd
# TODO: from collections import defaultdict
# TODO: from src.tc2_ecommerce.models.base import BaseRecommender
# TODO: 
# TODO: class ItemItemCF(BaseRecommender):
# TODO:     def __init__(self, config: dict):
# TODO:         self.config = config
# TODO:         self.co_visitation = defaultdict(dict)
# TODO:         self.trained = False
# TODO:     
# TODO:     def fit(self, train_data):
# TODO:         pass
# TODO:     
# TODO:     def predict(self, user_ids, k=10):
# TODO:         pass
