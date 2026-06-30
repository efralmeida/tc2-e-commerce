"""
Configuration module using Pydantic Settings.

Carrega variáveis de ambiente do arquivo .env ou variáveis de ambiente do sistema.
"""

from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """
    Settings para o projeto tc2-ecommerce.
    
    Carrega configurações de:
    1. Arquivo .env (local)
    2. Variáveis de ambiente do sistema (GitHub Actions, Docker, etc.)
    
    Ordem de precedência: variáveis de ambiente > .env > valores padrão
    """
    
    # ==========================================
    # MLflow Configuration
    # ==========================================
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_backend_store_uri: str = "file:./mlruns"
    mlflow_default_artifact_root: str = "s3://seu-bucket-mlflow/artifacts"
    mlflow_experiment_name: str = "baseline_experiments"
    
    # ==========================================
    # AWS Configuration (para S3)
    # ==========================================
    aws_access_key_id: Optional[str] = None       # carregado do sistema
    aws_secret_access_key: Optional[str] = None   # carregado do sistema
    aws_default_region: str = "us-east-1"
    
    # ==========================================
    # DVC Configuration
    # ==========================================
    dvc_remote_name: str = "local"
    dvc_remote_path: str = "./dvc_storage"
    
    # ==========================================
    # Data Configuration
    # ==========================================
    data_raw_path: str = "./data/raw"
    data_processed_path: str = "./data/processed"
    models_path: str = "./models"
    
    # ==========================================
    # Model Configuration
    # ==========================================
    top_k: int = 10
    max_candidates: int = 2000
    event_weights: str = "view:1.0,addtocart:3.0,transaction:5.0"
    
    # ==========================================
    # Baseline Hyperparameters
    # ==========================================
    baseline_max_items_by_popularity: int = 5000
    baseline_max_items_per_user: int = 30
    baseline_top_neighbors: int = 100
    
    # ==========================================
    # Neural Network Configuration
    # ==========================================
    neural_batch_size: int = 32
    neural_epochs: int = 50
    neural_learning_rate: float = 0.001
    neural_hidden_dim: int = 128
    
    # ==========================================
    # Logging Configuration
    # ==========================================
    log_level: str = "INFO"
    
    # ==========================================
    # Database Configuration (para futuro)
    # ==========================================
    db_host: Optional[str] = None
    db_port: Optional[int] = None
    db_name: Optional[str] = None
    db_user: Optional[str] = None
    db_password: Optional[str] = None
    
    # ==========================================
    # API Configuration (para futuro)
    # ==========================================
    api_host: str = "0.0.0.0"
    api_port: int = 5000
    api_debug: bool = False
    
    # ==========================================
    # Development Configuration
    # ==========================================
    environment: str = "development"
    
    class Config:
        """Configuração do Pydantic."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignora variáveis de ambiente extras
    
    def get_event_weights_dict(self) -> dict[str, float]:
        """
        Converte string de event_weights em dicionário.
        
        Exemplo:
            "view:1.0,addtocart:3.0,transaction:5.0" → 
            {"view": 1.0, "addtocart": 3.0, "transaction": 5.0}
        """
        weights = {}
        for pair in self.event_weights.split(","):
            event, weight = pair.strip().split(":")
            weights[event] = float(weight)
        return weights


# Instância global de configurações
settings = Settings()
