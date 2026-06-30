"""
Módulo de preparação de dados para DVC pipeline.

Funções para carregar dados brutos, aplicar split temporal e salvar processados.
"""
import json
from pathlib import Path

import pandas as pd


def prepare_data():
    """Preparar dados: load raw → split temporal → save processed."""
    
    # Configurações
    DATA_RAW_PATH = Path("data/raw/retailrocket/events.csv")
    DATA_PROCESSED_DIR = Path("data/processed")
    EVENT_WEIGHT = {"view": 1.0, "addtocart": 3.0, "transaction": 5.0}
    
    # Criar diretório processado
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Carregar dados brutos
    print(f"[DVC] Carregando dados de {DATA_RAW_PATH}...")
    df = pd.read_csv(DATA_RAW_PATH)
    
    # Filtrar por eventos válidos
    df = df[df["event"].isin(EVENT_WEIGHT.keys())].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["weight"] = df["event"].map(EVENT_WEIGHT).astype(float)
    
    # Ordenar por timestamp (crítico para split temporal)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df[["visitorid", "itemid"]] = df[["visitorid", "itemid"]].astype("int64")
    
    # Split temporal: 70/15/15
    n = len(df)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df = df.iloc[valid_end:].copy()
    
    # Salvar splits
    print(f"[DVC] Salvando dados processados em {DATA_PROCESSED_DIR}/...")
    train_df.to_csv(DATA_PROCESSED_DIR / "train.csv", index=False)
    valid_df.to_csv(DATA_PROCESSED_DIR / "valid.csv", index=False)
    test_df.to_csv(DATA_PROCESSED_DIR / "test.csv", index=False)
    
    # Salvar informações do split (para auditoria)
    split_info = {
        "total_interactions": len(df),
        "unique_users": df["visitorid"].nunique(),
        "unique_items": df["itemid"].nunique(),
        "train_size": len(train_df),
        "valid_size": len(valid_df),
        "test_size": len(test_df),
        "train_timestamp_range": [str(train_df["timestamp"].min()), str(train_df["timestamp"].max())],
        "valid_timestamp_range": [str(valid_df["timestamp"].min()), str(valid_df["timestamp"].max())],
        "test_timestamp_range": [str(test_df["timestamp"].min()), str(test_df["timestamp"].max())],
        "event_weights": EVENT_WEIGHT,
    }
    
    with open(DATA_PROCESSED_DIR / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    
    print("[DVC] ✅ Dados preparados e salvos com sucesso!")
    print(f"  - Train: {len(train_df)} interações")
    print(f"  - Valid: {len(valid_df)} interações")
    print(f"  - Test: {len(test_df)} interações")


if __name__ == "__main__":
    prepare_data()
