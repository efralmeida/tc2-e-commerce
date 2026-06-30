"""Estrategias de preprocessamento para o pipeline de recomendacao.

Este modulo implementa o padrao Strategy: cada classe representa uma forma
de transformar os dados antes do treino.

A ideia e separar o "como transformar" do resto do pipeline. Assim, trocar
de preprocessamento vira apenas trocar a estrategia escolhida.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
import pandas as pd


class BasePreprocessor(ABC):
  """Interface base para qualquer estrategia de preprocessamento.

  Contrato esperado:
  - fit: aprende parametros a partir dos dados (quando necessario);
  - transform: aplica a transformacao nos dados;
  - fit_transform: atalho para fit + transform.
  """

  name: str = "base"

  @abstractmethod
  def fit(self, data: pd.DataFrame) -> "BasePreprocessor":
    """Aprende parametros internos da transformacao."""

  @abstractmethod
  def transform(self, data: pd.DataFrame) -> pd.DataFrame:
    """Aplica transformacao nos dados e retorna um novo DataFrame."""

  def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
    """Executa fit seguido de transform para reduzir boilerplate."""

    self.fit(data)
    return self.transform(data)

  def _validate_data(self, data: pd.DataFrame) -> None:
    """Valida formato minimo esperado para evitar falhas silenciosas."""

    if not isinstance(data, pd.DataFrame):
      raise ValueError("O preprocessor espera um pandas.DataFrame.")

    if data.empty:
      raise ValueError("DataFrame vazio: nao ha dados para preprocessar.")


class StandardPreprocessor(BasePreprocessor):
  """Estrategia baseline: valida e retorna copia dos dados.

  Essa estrategia e util quando os dados ja estao no formato correto e
  voce quer explicitar no pipeline que nao havera transformacao adicional.
  """

  name = "standard"

  def fit(self, data: pd.DataFrame) -> "StandardPreprocessor":
    self._validate_data(data)
    return self

  def transform(self, data: pd.DataFrame) -> pd.DataFrame:
    self._validate_data(data)
    return data.copy()


class NormalizationPreprocessor(BasePreprocessor):
  """Normaliza colunas numericas para o intervalo [0, 1].

  Regras adotadas:
  - apenas colunas numericas entram na normalizacao;
  - colunas constantes viram 0.0 (evita divisao por zero);
  - colunas nao numericas sao preservadas sem alteracao.
  """

  name = "normalization"

  def __init__(self, exclude_columns: list[str] | None = None):
    self.exclude_columns = set(exclude_columns or [])
    self._numeric_columns: list[str] = []
    self._min_values: dict[str, float] = {}
    self._max_values: dict[str, float] = {}

  def fit(self, data: pd.DataFrame) -> "NormalizationPreprocessor":
    self._validate_data(data)

    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    self._numeric_columns = [
      col for col in numeric_columns if col not in self.exclude_columns
    ]

    for column in self._numeric_columns:
      # Guardamos min/max no fit para aplicar exatamente o mesmo scaling
      # em validacao e teste, evitando data leakage.
      self._min_values[column] = float(data[column].min())
      self._max_values[column] = float(data[column].max())

    return self

  def transform(self, data: pd.DataFrame) -> pd.DataFrame:
    self._validate_data(data)

    transformed = data.copy()

    for column in self._numeric_columns:
      min_value = self._min_values[column]
      max_value = self._max_values[column]
      denominator = max_value - min_value

      if denominator == 0:
        transformed[column] = 0.0
      else:
        transformed[column] = (transformed[column] - min_value) / denominator

    return transformed


class InteractionWeightingPreprocessor(BasePreprocessor):
  """Converte tipo de evento em peso numerico para recomendacao.

  Default de pesos:
  - view: 1.0
  - addtocart: 3.0
  - transaction: 5.0
  """

  name = "interaction_weighting"

  def __init__(
    self,
    event_column: str = "event",
    output_column: str = "interaction_weight",
    event_weights: Mapping[str, float] | None = None,
  ):
    self.event_column = event_column
    self.output_column = output_column
    self.event_weights = dict(
      event_weights
      or {
        "view": 1.0,
        "addtocart": 3.0,
        "transaction": 5.0,
      }
    )

  def fit(self, data: pd.DataFrame) -> "InteractionWeightingPreprocessor":
    self._validate_data(data)
    self._validate_event_column(data)
    return self

  def transform(self, data: pd.DataFrame) -> pd.DataFrame:
    self._validate_data(data)
    self._validate_event_column(data)

    transformed = data.copy()
    transformed[self.output_column] = (
      transformed[self.event_column]
      .map(self.event_weights)
      .fillna(1.0)
      .astype(float)
    )

    return transformed

  def _validate_event_column(self, data: pd.DataFrame) -> None:
    if self.event_column not in data.columns:
      raise ValueError(
        f"Coluna de evento '{self.event_column}' nao encontrada no DataFrame."
      )


class PreprocessorFactory:
  """Factory de estrategias de preprocessamento.

  Pode ser usada para criar preprocessadores por nome no pipeline e para
  registrar novas estrategias sem alterar codigo existente (open/closed).
  """

  _registry: ClassVar[dict[str, type[BasePreprocessor]]] = {
    "standard": StandardPreprocessor,
    "normalization": NormalizationPreprocessor,
    "interaction_weighting": InteractionWeightingPreprocessor,
  }

  @classmethod
  def create(
    cls,
    name: str,
    config: Mapping[str, Any] | None = None,
  ) -> BasePreprocessor:
    """Cria uma estrategia pelo nome, usando config opcional."""

    normalized_name = cls._normalize_name(name)

    if normalized_name not in cls._registry:
      available = ", ".join(cls.available_strategies())
      raise ValueError(
        f"Estrategia desconhecida: '{normalized_name}'. "
        f"Disponiveis: {available}."
      )

    strategy_class = cls._registry[normalized_name]
    return strategy_class(**dict(config or {}))

  @classmethod
  def register(
    cls,
    name: str,
    strategy_class: type[BasePreprocessor],
    *,
    overwrite: bool = False,
  ) -> None:
    """Registra uma nova estrategia no factory."""

    normalized_name = cls._normalize_name(name)

    if not normalized_name:
      raise ValueError("O nome da estrategia nao pode ser vazio.")

    if not issubclass(strategy_class, BasePreprocessor):
      raise ValueError(
        "A estrategia registrada deve herdar de BasePreprocessor."
      )

    if not overwrite and normalized_name in cls._registry:
      raise ValueError(
        f"A estrategia '{normalized_name}' ja esta registrada."
      )

    cls._registry[normalized_name] = strategy_class

  @classmethod
  def available_strategies(cls) -> tuple[str, ...]:
    """Retorna lista ordenada das estrategias registradas."""

    return tuple(sorted(cls._registry))

  @staticmethod
  def _normalize_name(name: str) -> str:
    return name.strip().lower()
