# tc2-e-commerce

Projeto de recomendacao para o Tech Challenge 02 com pipeline de dados, baselines,
modelo neural em PyTorch, DVC e MLflow.

## Estrutura principal

- src/tc2_ecommerce/models: implementacoes de Dummy, Item-Item e Neural.
- src/tc2_ecommerce/train.py: treino e grid search.
- src/tc2_ecommerce/app.py: orquestracao de alto nivel.
- src/tc2_ecommerce/api.py: endpoints de servico e handler AWS Lambda.
- notebook/baseline_experiments.ipynb: experimentos exploratorios e comparacoes.

## Setup rapido

1. Instalar dependencias:

```bash
poetry install
```

2. Copiar variaveis de ambiente:

```bash
cp .env.exemplo .env
```

3. Preparar dados (se necessario):

```bash
poetry run python -m tc2_ecommerce.data
```

## Treino e avaliacao

Treinar baseline dummy:

```bash
poetry run python -m tc2_ecommerce.main --action train --model dummy
```

Avaliar modelo salvo:

```bash
poetry run python -m tc2_ecommerce.main --action evaluate --model dummy
```

## API local e AWS

O modulo api.py suporta:

- uso local via funcao create_fastapi_app() (se FastAPI estiver instalado);
- uso em AWS Lambda via lambda_handler (API Gateway proxy).

Exemplo de roteamento Lambda:

- GET /health
- POST /recommend
- POST /evaluate

## Docker

Build da imagem:

```bash
docker build -t tc2-ecommerce:latest .
```

Executar:

```bash
docker run --rm -p 5000:5000 --env-file .env tc2-ecommerce:latest
```

## Observacoes

- Para API HTTP completa, instale FastAPI e Uvicorn:

```bash
poetry add --group dev fastapi uvicorn
```

- Para reproducibilidade, registre seed e metadados com reproducibility.py.