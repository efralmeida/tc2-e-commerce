"""
Factory para criar modelos de recomendação de forma centralizada.

O objetivo deste módulo é concentrar a criação dos modelos em um único ponto.
Isso facilita três coisas importantes:

1. Trocar o modelo central sem espalhar `if/elif` pelo código.
2. Registrar novos modelos dinamicamente sem alterar o fluxo principal.
3. Evitar importações pesadas no carregamento do pacote, algo relevante para
   modelos neurais que podem depender de bibliotecas opcionais como PyTorch.

A implementação abaixo usa o padrão Factory com um registro interno.
Cada entrada do registro associa um nome amigável, como `dummy` ou
`item_item`, a uma função construtora que recebe um dicionário de config.

Isso é útil no projeto porque o módulo ainda está evoluindo: a factory pode
existir agora, mesmo antes de os modelos finais estarem prontos.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, ClassVar


logger = logging.getLogger(__name__)


ModelBuilder = Callable[[Mapping[str, Any]], Any]


class ModelFactory:
  """Cria e registra instâncias de modelos de recomendação.

  A classe mantém um registro em memória com os construtores disponíveis.
  O método :meth:`create` resolve o nome do modelo, prepara a configuração e
  delega a criação para o construtor correto.

  O comportamento é propositalmente simples:
  - nomes de modelo são normalizados para minúsculas e sem espaços externos;
  - a configuração é copiada antes de ser entregue ao modelo;
  - erros são explícitos, com mensagem contendo os nomes disponíveis.
  """

  # Registro central de construtores. A chave é o nome público do modelo.
  _registry: ClassVar[dict[str, ModelBuilder]] = {}
  _defaults_loaded: ClassVar[bool] = False

  @classmethod
  def register(
    cls,
    name: str,
    builder: ModelBuilder,
    *,
    overwrite: bool = False,
  ) -> None:
    """Registra um novo modelo no factory.

    Args:
      name: Nome público do modelo, por exemplo `dummy`.
      builder: Função ou classe que cria a instância do modelo.
      overwrite: Permite substituir uma entrada já existente.

    Raises:
      ValueError: Se o nome for vazio, o builder for inválido ou houver
        conflito com um registro existente sem `overwrite=True`.
    """

    normalized_name = cls._normalize_name(name)

    if not normalized_name:
      raise ValueError("O nome do modelo não pode ser vazio.")

    if not callable(builder):
      raise ValueError("O builder informado não é chamável.")

    if not overwrite and normalized_name in cls._registry:
      raise ValueError(
        f"O modelo '{normalized_name}' já está registrado. "
        "Use overwrite=True para substituir a entrada."
      )

    cls._registry[normalized_name] = builder
    logger.debug("Modelo '%s' registrado no ModelFactory.", normalized_name)

  @classmethod
  def create(
    cls,
    model_name: str,
    config: Mapping[str, Any] | None = None,
  ) -> Any:
    """Cria uma instância do modelo solicitado.

    Args:
      model_name: Nome do modelo a ser criado.
      config: Configuração do modelo. Se `None`, usa um dicionário vazio.

    Returns:
      Instância do modelo correspondente ao nome informado.

    Raises:
      ValueError: Se o nome do modelo não estiver registrado.
      RuntimeError: Se o construtor registrado lançar uma exceção durante
        a criação da instância.
    """

    normalized_name = cls._normalize_name(model_name)

    if not normalized_name:
      raise ValueError("O nome do modelo não pode ser vazio.")

    cls._ensure_default_models()

    try:
      builder = cls._registry[normalized_name]
    except KeyError as exc:
      available_models = ", ".join(cls.available_models()) or "nenhum"
      raise ValueError(
        f"Modelo desconhecido: '{normalized_name}'. "
        f"Modelos disponíveis: {available_models}."
      ) from exc

    normalized_config = dict(config or {})

    try:
      instance = builder(normalized_config)
    except Exception as exc:  # pragma: no cover - reempacotamento de erro
      logger.exception(
        "Falha ao criar o modelo '%s' com a configuração recebida.",
        normalized_name,
      )
      raise RuntimeError(
        f"Não foi possível criar o modelo '{normalized_name}'."
      ) from exc

    logger.info("Modelo '%s' criado com sucesso.", normalized_name)
    return instance

  @classmethod
  def available_models(cls) -> tuple[str, ...]:
    """Retorna os nomes dos modelos atualmente disponíveis."""

    cls._ensure_default_models()
    return tuple(sorted(cls._registry))

  @classmethod
  def _ensure_default_models(cls) -> None:
    """Registra os modelos padrão do projeto, se possível.

    As importações são feitas sob demanda para evitar que o pacote inteiro
    falhe ao importar caso algum modelo ainda não esteja implementado ou
    dependa de uma biblioteca opcional que não esteja instalada.
    """

    if cls._defaults_loaded:
      return

    cls._register_default(
      name="dummy",
      import_path="tc2_ecommerce.models.dummy",
      class_name="DummyBaseline",
    )
    cls._register_default(
      name="item_item",
      import_path="tc2_ecommerce.models.item_item",
      class_name="ItemItemCF",
    )
    cls._register_default(
      name="neural",
      import_path="tc2_ecommerce.models.neural",
      class_name="NeuralRecommender",
    )

    cls._defaults_loaded = True

  @classmethod
  def _register_default(
    cls,
    *,
    name: str,
    import_path: str,
    class_name: str,
  ) -> None:
    """Cria um builder lazy para um modelo padrão.

    O builder só importa a classe do modelo quando o modelo for solicitado.
    Isso evita acoplamento forte com implementações que ainda estão em
    desenvolvimento e reduz o custo de importação do pacote.
    """

    def builder(config: Mapping[str, Any]) -> Any:
      module = __import__(import_path, fromlist=[class_name])
      model_class = getattr(module, class_name)
      return model_class(dict(config))

    try:
      cls.register(name, builder)
    except ValueError:
      # Se o nome já estiver registrado por código externo, respeitamos a
      # decisão do chamador e mantemos o comportamento atual.
      logger.debug("Modelo padrão '%s' já estava registrado.", name)

  @staticmethod
  def _normalize_name(name: str) -> str:
    """Normaliza o nome do modelo para reduzir erros de uso."""

    return name.strip().lower()
