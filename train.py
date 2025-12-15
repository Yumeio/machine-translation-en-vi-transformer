import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

from config import get_config, get_weights_file_path, latest_weights_file_path
from data_loader import get_dataloaders
from model import Transformer
from optim import AdamW, Adam
from lr_scheduler import LRScheduler
from utils import get_device, set_seed

def get_model(config, src_vocab_size, tgt_vocab_size):
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
    )
    return model

def train_model(config):
    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Make sure model folder exists
    Path(config.training.model_folder).mkdir(parents=True, exist_ok=True)

    # 1. Get DataLoaders and Tokenizers
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataloaders(config)
    
    src_vocab_size = tokenizer_src.get_vocab_size()
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()
    
    # Update config with vocab sizes (optional, mostly for checking)
    config.model.src_vocab_size = src_vocab_size
    config.model.tgt_vocab_size = tgt_vocab_size

    # 2. Build Model
    model = get_model(config, src_vocab_size, tgt_vocab_size).to(device)

    # 3. Optimizer
    optim_cls = AdamW if config.training.optim_type == "adamw" else Adam 
    optimizer = optim_cls(
        model.parameters(), 
        lr=config.training.lr, 
        betas=(0.9, 0.98), 
        eps=1e-9
    )
    
    # 3.1 LR Scheduler
    scheduler = LRScheduler(
        optimizer, 
        d_model=config.model.d_model, 
        warmup_steps=4000
    )

    # 4. Load Pretrained Weights if specified
    initial_epoch = 0
    global_step = 0
    preload = config.training.preload
    
    if preload:
        model_filename = get_weights_file_path(
            config, preload
        ) if preload != 'latest' else latest_weights_file_path(config)
        
        if model_filename:
            print(f"Preloading model {model_filename}")
            state = torch.load(model_filename)
            initial_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            # Note: Scheduler state loading is not implemented in standard flow here but could be added
            global_step = state['global_step']
            del state
        else:
            print(f"Could not find model to preload: {preload}. Starting from scratch.")

    # 5. Loss Function
    # Ignore padding token in loss calculation
    pad_token_id = tokenizer_tgt.token_to_id("[PAD]")
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1).to(device)

    # 6. Tensorboard
    writer = SummaryWriter(config.training.experiment_name)

    # 7. Training Loop
    for epoch in range(initial_epoch, config.training.num_epochs):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (B, Seq_Len)
            decoder_input = batch['decoder_input'].to(device) # (B, Seq_Len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, Seq_Len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, Seq_Len, Seq_Len)
            
            # (B, Seq_Len) -> Expected Output
            label = batch['decoder_output'].to(device) 

            # Forward pass
            # (B, Seq_Len, Vocab_Size)
            proj_output = model(encoder_input, decoder_input, encoder_mask, decoder_mask) 

            # Compute Loss
            # View: (B * Seq_Len, Vocab_Size) vs (B * Seq_Len)
            loss = loss_fn(proj_output.view(-1, tgt_vocab_size), label.view(-1))
            
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

            # Log to tensorboard
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            writer.flush()

            # Backprop
            loss.backward()
            optimizer.step()
            scheduler.step() # Update LR
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Validation at end of epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config.model.max_seq_len, device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save model
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

def run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer):
    model.eval()
    count = 0
    
    # We can just show a few examples or compute loss. 
    # Let's compute validation loss first.
    # Re-instantiate loss function for validation just to be safe it's clean or same inst
    pad_token_id = tokenizer_tgt.token_to_id("[PAD]")
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1).to(device)
    
    total_val_loss = 0
    num_batches = 0

    # Console width
    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80
        
    with torch.no_grad():
        for batch in val_dataloader:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['decoder_output'].to(device)

            proj_output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
            
            # Compute Val Loss
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            total_val_loss += loss.item()
            num_batches += 1
            
            # Print Examples (limit to 2 per validation run)
            # This requires Greedy Decode or Beam Search (not implemented in model directly yet)
            # For now, just logging loss is better than failing if greedy_decode isn't ready.
            # But the user might want translation examples.
            
    if num_batches > 0:
        avg_val_loss = total_val_loss / num_batches
        writer.add_scalar('val_loss', avg_val_loss, global_step)
        writer.flush()
        print_msg(f"Validation Loss: {avg_val_loss:.3f}")

if __name__ == '__main__':
    config = get_config()
    train_model(config)
