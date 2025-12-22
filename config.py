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
    num_heads: int = 4         
    d_ff: int = (512 * 4)          
    num_encoder_blocks: int = 6
    num_decoder_blocks: int = 6  
    dropout: float = 0.1       
    activation: Literal['relu', 'gelu'] = "relu"
    max_seq_len: int = 128     
    src_vocab_size: Optional[int] = 30000 
    tgt_vocab_size: Optional[int] = 30000 
    tie_weights: bool = True

@dataclass
class TrainingConfig:
    batch_size: int = 64 
    num_workers: int = 4
    prefetch_factor: int = 2
    num_epochs: int = 20    
    lr: float = 3e-4            
    weight_decay: float = 0.01
    optim_type: Literal["adam", "adamw"] = "adamw"
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.1
    warmup_steps: int = 4000
    model_folder: str = "weights"
    model_basename: str = "tmodel_"
    preload: Optional[str] = None
    experiment_name: str = "runs/tmodel"
    eval_steps: int = 1000
    save_steps: int = 5
    use_amp: bool = False

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    dataset: DatasetConfig

def get_config(config_path: str = "config.json") -> Config:
    if Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return Config(
                model=ModelConfig(**config_data.get('model', {})),
                training=TrainingConfig(**config_data.get('training', {})),
                dataset=DatasetConfig(**config_data.get('dataset', {}))
            )
        except Exception:
            pass
    
    return Config(model=ModelConfig(), training=TrainingConfig(), dataset=DatasetConfig())

def save_config(config: Config, config_path: str = "config.json"):
    config_dict = {
        "model": vars(config.model),
        "training": vars(config.training),
        "dataset": vars(config.dataset)
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

def get_weights_file_path(config: Config, epoch: str):
    model_folder = config.training.model_folder
    model_basename = config.training.model_basename
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def get_best_model(config: Config):
    model_folder = config.training.model_folder
    model_basename = config.training.model_basename
    model_filename = f"best_model.pt"
    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(config: Config):
    model_folder = config.training.model_folder
    model_basename = config.training.model_basename
    model_filename = f"{model_basename}*.pt"
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    print(f"Latest weights found: {weights_files[len(weights_files) - 1]}")
    return str(weights_files[len(weights_files) - 1])
