# DVC + MLflow Workflow

Este documento explica como usar DVC e MLflow juntos para reprodutibilidade e rastreamento de experimentos.

O fluxo abaixo cobre tanto métricas exploratórias em notebook quanto rastreamento para pipeline de produção.

## 🎯 Objetivo

- **MLflow**: Rastrear métricas, parâmetros e modelos durante experimentos (notebooks)
- **DVC**: Versioncar dados, processados e artefatos para garantir reprodutibilidade completa

## 📊 Arquitetura

```
data/raw/retailrocket/*.csv (dados brutos - versionados com DVC)
        ↓
src/tc2_ecommerce/data.py (prepare_data)
        ↓
data/processed/{train,valid,test}.csv (dados processados - artefatos DVC)
        ↓
notebook/baseline_experiments.ipynb (MLflow rastreia métricas)
        ↓
models/{dummy, baseline_item_item, neural}_*.json
        ↓
mlruns/ (experimentos MLflow)
dvc.lock (reprodutibilidade DVC)
```

## 🚀 Como usar

### 1. Preparar dados (via DVC)

```bash
# Opção 1: Manualmente
python -c "from src.tc2_ecommerce.data import prepare_data; prepare_data()"

# Opção 2: Via DVC (reproduz exatamente)
dvc repro data://prepare
```

### 2. Executar experimentos (via notebook + MLflow)

```bash
# Abrir notebook e executar cells (MLflow rastreia automaticamente)
jupyter notebook notebook/baseline_experiments.ipynb

# Ou executar via CLI
jupyter nbconvert --to notebook --execute notebook/baseline_experiments.ipynb
```

### 3. Visualizar experimentos

```bash
# MLflow UI (métricas em tempo real)
mlflow ui
# Acessa http://localhost:5000

# DVC status (reprodutibilidade)
dvc status
```

### 4. Reproduzir pipeline completo

```bash
# Rodar todo o pipeline determinístico
dvc repro

# Ver o que vai ser executado
dvc dag

# Limpar e refazer
dvc repro --force
```

## 📝 Etapas e artefatos

| Etapa | Input | Output | Rastreamento |
|-------|-------|--------|--------------|
| `prepare` | `data/raw/retailrocket/*.csv` | `data/processed/{train,valid,test}.csv` | DVC |
| `baseline_experiments` | `data/processed/*.csv` | `models/{recs,metrics}.json` | DVC + MLflow |
| `neural_experiments` (future) | `data/processed/*.csv` | `models/neural_*.pth` | DVC + MLflow |

## 🔄 Reprodutibilidade

**Com DVC**: Alguém pode fazer apenas:
```bash
dvc pull    # Baixar dados e modelos
dvc repro   # Reproduzir pipeline completo
```

**Com MLflow**: Acompanhar cada experimento:
```bash
mlflow experiments search
mlflow runs list --experiment-id 0
```

## 🛠️ Adicionando novos stages

Adicione em `dvc.yaml`:

```yaml
stages:
  new_stage:
    cmd: python -m src.tc2_ecommerce.my_module arg1 arg2
    deps:
      - data/processed/train.csv
    outs:
      - models/new_output.pkl
    metrics:
      - metrics.json:
          cache: false
```

Depois rodar:
```bash
dvc repro
```

## 📚 Recursos

- [DVC Docs](https://dvc.org/doc)
- [MLflow Docs](https://mlflow.org/docs)
- [DVC Pipeline](https://dvc.org/doc/user-guide/pipelines)
