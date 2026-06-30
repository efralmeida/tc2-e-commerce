"""
Classe abstrata para modelos de recomendação.

Define interface comum que todos os modelos devem implementar.

TODO: Implementar classe BaseRecommender (ABC):
  - fit(train_data) → treinar modelo
  - predict(user_ids, k=10) → retornar top-k recommendations
  - evaluate(test_data) → calcular métricas
  - save(path) → serializar modelo
  - load(path) → desserializar modelo

TODO: Adicionar propriedades comuns:
  - name: str (identificador do modelo)
  - version: str (versão do modelo)
  - config: dict (hiperparâmetros)
  - trained: bool (se foi treinado)

TODO: Implementar métodos auxiliares:
  - _validate_input(data) → validar formato de entrada
  - _log_params(mlflow_client) → registrar hiperparâmetros no MLflow
  - _log_metrics(mlflow_client, metrics) → registrar métricas

TODO: Adicionar type hints robustos
TODO: Adicionar docstrings detalhadas
TODO: Adicionar validação de estado (ex: só fazer predict se treinado)
"""

# TODO: from abc import ABC, abstractmethod
# TODO: from typing import Dict, List, Tuple, Optional
# TODO: import pandas as pd
# TODO: 
# TODO: class BaseRecommender(ABC):
# TODO:     @abstractmethod
# TODO:     def fit(self, train_data):
# TODO:         pass
# TODO:     
# TODO:     @abstractmethod
# TODO:     def predict(self, user_ids, k=10):
# TODO:         pass
# TODO:     
# TODO:     @abstractmethod
# TODO:     def evaluate(self, test_data):
# TODO:         pass
