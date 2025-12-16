from dataclasses import dataclass
from typing import Literal, Optional
from pathlib import Path

import json

@dataclass
class DatasetConfig:
    train_path: str = "./dataset/processed/train.parquet"
    validation_path: str = "./dataset/processed/validation.parquet"
    src_lang: str = "en"
    tgt_lang: str = "vi"
    tokenizer_file: str = "tokenizer_{0}.json"

@dataclass
class ModelConfig:
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 512 * 4
    activation: Literal['relu', 'gelu'] = "gelu"
    dropout: float = 0.1
    num_encoder_blocks: int = 6
    num_decoder_blocks: int = 6
    max_seq_len: int = 350
    src_vocab_size: Optional[int] = 30000 
    tgt_vocab_size: Optional[int] = 30000 

@dataclass
class TrainingConfig:
    batch_size: int = 16
    num_workers: int = 4
    num_epochs: int = 20
    lr: float = 4e-4
    weight_decay: float = 1e-5
    optim_type: Literal["adam", "adamw"] = "adamw"
    model_folder: str = "weights"
    model_basename: str = "tmodel_"
    preload: Optional[str] = None
    experiment_name: str = "runs/tmodel"
@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    dataset: DatasetConfig

def get_config(config_path: str = "best_config.json") -> Config:
    if Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            print(f"Loading config from {config_path}")
            return Config(
                model=ModelConfig(**config_data.get('model', {})),
                training=TrainingConfig(**config_data.get('training', {})),
                dataset=DatasetConfig(**config_data.get('dataset', {}))
            )
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}.  Using default config.")
    
    return Config(
        model=ModelConfig(),
        training=TrainingConfig(),
        dataset=DatasetConfig()
    )

def get_weights_file_path(config: Config, epoch: str):
    model_folder = config.training.model_folder
    model_basename = config.training.model_basename
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config: Config):
    model_folder = config.training.model_folder
    model_basename = config.training.model_basename
    model_filename = f"{model_basename}*.pt"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])