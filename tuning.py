import torch
import torch.nn as nn
import optuna
import time
import json
import dataclasses
from pathlib import Path
from tqdm import tqdm

from config import get_config, Config
from data_loader import get_dataloaders
from model import Transformer
from optim import AdamW 
from lr_scheduler import LRScheduler
from utils import get_device, set_seed

def objective(trial):
    torch.cuda.empty_cache() # Clear cache before starting a new trial
    
    # 1. Suggest Hyperparameters
    config = get_config()
    
    # Model params
    d_model_heads_map = {
        256: [4, 8],
        512: [8],
    }
    
    config.model.d_model = trial.suggest_categorical("d_model", [256, 512])
    possible_heads = d_model_heads_map[config.model.d_model]
    config.model.num_heads = trial.suggest_categorical("num_heads", possible_heads)
    
    config.model.d_ff = trial.suggest_categorical("d_ff", [1024, 2048])
    config.model.dropout = trial.suggest_float("dropout", 0.1, 0.3)
    
    # Training params
    config.training.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    config.training.batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    config.training.num_epochs = 3 # Run for fewer epochs for tuning
    
    # Overwrite experiment name
    config.training.experiment_name = f"runs/tuning_trial_{trial.number}"
    
    device = get_device()
    set_seed(42)

    # 2. Prepare Data
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataloaders(config)
    src_vocab_size = tokenizer_src.get_vocab_size()
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()

    # 3. Model
    model = Transformer(
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        activation=config.model.activation,
        dropout=config.model.dropout,
        num_encoder_blocks=config.model.num_encoder_blocks,
        num_decoder_blocks=config.model.num_decoder_blocks,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_seq_len=config.model.max_seq_len
    ).to(device)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), 
        lr=config.training.lr, 
        betas=(0.9, 0.98), 
        eps=1e-9
    )
    
    scheduler = LRScheduler(
        optimizer, 
        d_model=config.model.d_model, 
        warmup_steps=1000 # Reduced warmup for shorter tuning
    )

    # 5. Loss
    pad_token_id = tokenizer_tgt.token_to_id("[PAD]")
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1).to(device)

    # 6. Training Loop (Simplified)
    for epoch in range(config.training.num_epochs):
        start_time = time.time()
        model.train()
        batch_iterator = tqdm(
            train_dataloader, 
            desc=f"Trial {trial.number} | Epoch {epoch+1}/{config.training.num_epochs}",
            mininterval=1,
            leave=False,
            dynamic_ncols=True
        )
        
        train_loss_accum = 0
        num_train_batches = 0
        
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['decoder_output'].to(device)

            proj_output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
            loss = loss_fn(proj_output.view(-1, tgt_vocab_size), label.view(-1))
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss_accum += loss.item()
            num_train_batches += 1
            batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss_accum / num_train_batches

        # Validation
        model.eval()
        val_loss_accum = 0.0
        num_val_batches = 0
        
        # Validation loop
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation", leave=False, mininterval=1, dynamic_ncols=True):
                encoder_input = batch['encoder_input'].to(device)
                decoder_input = batch['decoder_input'].to(device)
                encoder_mask = batch['encoder_mask'].to(device)
                decoder_mask = batch['decoder_mask'].to(device)
                label = batch['decoder_output'].to(device)

                proj_output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
                loss = loss_fn(proj_output.view(-1, tgt_vocab_size), label.view(-1))
                val_loss_accum += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss_accum / num_val_batches
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        
        print(f"Trial {trial.number} | Epoch {epoch+1}: "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Time: {epoch_duration:.2f}s")

        # Report to Optuna
        trial.report(avg_val_loss, epoch)

        # Pruning
        if trial.should_prune():
            raise optuna.TrialPruned()

    return avg_val_loss

def save_config_tuning(config: Config, path: str):
    config_dict = dataclasses.asdict(config)
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"Config saved to {path}")

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10, show_progress_bar=True)

    print("Best trials:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("Best Params:", study.best_params)
    print("Best Loss:", study.best_value)
    
    # Generate config.json for the best model
    best_config = get_config()
    
    # Update config with best params
    best_config.model.d_model = trial.params["d_model"]
    best_config.model.num_heads = trial.params["num_heads"]
    best_config.model.d_ff = trial.params["d_ff"]
    best_config.model.dropout = trial.params["dropout"]
    
    best_config.training.lr = trial.params["lr"]
    best_config.training.batch_size = trial.params["batch_size"]
    
    # Save the config
    save_config_tuning(best_config, "best_config.json")