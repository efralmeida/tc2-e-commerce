"""
Pipeline de treinamento de modelos.

Orquestra: preparação dados → treinamento → avaliação → logging MLflow

TODO: Implementar função train_model():
  - Receber model_name, config, train_data, valid_data
  - Inicializar modelo via ModelFactory
  - Executar fit(train_data)
  - Avaliar em valid_data
  - Log de métricas no MLflow
  - Salvar modelo em models/

TODO: Implementar grid_search():
  - Receber model_name, hyperparameter_grid
  - Loop sobre combinações de hiperparâmetros
  - Treinar + avaliar para cada combo
  - Log de cada run no MLflow
  - Retornar melhor configuração (por métrica especificada)

TODO: Implementar cross_validation():
  - k-fold validation para avaliação mais robusta
  - Integrar com MLflow (salvar múltiplos runs)

TODO: Integração com reproducibility:
  - Chamar set_seed() no início
  - Log de system info + versions

TODO: Integração com logging_utils:
  - Log de progresso durante treinamento
  - Log de tempo de execução
  - Log de memória utilizada

TODO: Tratamento de erros robusto (timeout, OOM, etc.)
TODO: Adicionar early stopping para neural networks
"""

# TODO: from src.tc2_ecommerce.factories.model_factory import ModelFactory
# TODO: from src.tc2_ecommerce.evaluation import Evaluator
# TODO: from src.tc2_ecommerce.reproducibility import set_seed
# TODO: import mlflow
# TODO: Implementar funções aqui
