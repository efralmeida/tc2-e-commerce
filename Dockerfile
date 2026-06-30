# TODO: Usar multi-stage build para reduzir tamanho da imagem
# TODO: Adicionar usuário non-root por questões de segurança
# TODO: Adicionar health check
# TODO: Configurar cache de layers do Docker para otimizar rebuilds

FROM python:3.12-slim

WORKDIR /app

# TODO: Instalar apenas dependências de produção (remover dev dependencies)
# Instalar poetry
RUN pip install --no-cache-dir poetry==1.8.0

# Copiar pyproject.toml e poetry.lock
COPY pyproject.toml poetry.lock ./

# Instalar dependências do projeto
# TODO: Usar --no-dev para não instalar dependências de desenvolvimento em produção
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Copiar código do projeto
COPY src/ ./src/
COPY notebook/ ./notebook/
COPY data/ ./data/
COPY configs/ ./configs/
COPY dvc.yaml ./dvc.yaml

# TODO: Configurar comando padrão (rodar baseline, neural, ou ambos?)
# TODO: Adicionar suporte para argumentos (CMD vs ENTRYPOINT)
CMD ["python", "-m", "tc2_ecommerce.main"]

# TODO: Expor portas se necessário (MLflow UI, Jupyter, API, etc.)
# EXPOSE 5000 5001 8888
