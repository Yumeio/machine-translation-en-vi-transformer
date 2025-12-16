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
import matplotlib.pyplot as plt
from metric import get_metric
from infer import greedy_decode, beam_search_decode

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
    print(f"Number of parameters: {model._sum_parameter()}")

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
    train_losses = []
    val_losses = []
    val_bleus = []
    
    # Placeholders for last validation result
    last_predicted = []
    last_expected = []
    last_source = []

    for epoch in range(initial_epoch, config.training.num_epochs):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}", ascii=True, ncols=100)
        
        epoch_loss = 0
        num_batches = 0 
        
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

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)

        # Validation at end of epoch
        val_loss, val_bleu, last_predicted, last_expected, last_source = run_validation(
            model, val_dataloader, tokenizer_src, tokenizer_tgt, config.model.max_seq_len, device, 
            lambda msg: batch_iterator.write(msg), global_step, writer
        )
        val_losses.append(val_loss)
        val_bleus.append(val_bleu)
        batch_iterator.write(f"Epoch {epoch:02d} Summary | Validation Loss: {val_loss:.3f} | BLEU: {val_bleu:.4f}")


        # Save model
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

    # Plotting Loss
    matplotlib.use('Agg')
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_plot.png')
    print("Saved loss plot to loss_plot.png")
    
    # Plotting BLEU
    plt.figure(figsize=(10, 5))
    plt.plot(val_bleus, label='Validation BLEU')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.legend()
    plt.title('Validation BLEU Score')
    plt.savefig('bleu_plot.png')
    print("Saved BLEU plot to bleu_plot.png")


    # Final Metrics Reporting
    print(f"Final Validation BLEU: {val_bleus[-1] if val_bleus else 0.0}")

def run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer):
    model.eval()
    count = 0
    
    pad_token_id = tokenizer_tgt.token_to_id("[PAD]")
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1).to(device)
    
    total_val_loss = 0
    num_batches = 0
    
    source_texts = []
    expected = []
    predicted = []

    # Console width
    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80
        
    with torch.no_grad():
        for batch in val_dataloader:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['decoder_output'].to(device)

            proj_output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
            
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            total_val_loss += loss.item()
            num_batches += 1
            
            src_text = batch['src_texts']
            tgt_text = batch['tgt_texts']
            source_texts.extend(src_text)
            expected.extend(tgt_text)
            
            # Batch decoding for BLEU (decoding one by one is slow but simple)
            # greedy_decode expects single item source, so we iterate
            
            for i in range(encoder_input.size(0)):
                source = encoder_input[i].contiguous().view(1, -1)
                source_mask = encoder_mask[i].contiguous() # (1, 1, seq_len)
                
                # Check dimensions of source_mask, greedy_decode expects (1, 1, 1, seq_len) or similar? 
                # greedy_decode uses it in encode: model.encode(source, source_mask)
                # In train loop: encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, Seq_Len)
                # So here: encoder_mask[i] is (1, 1, Seq_Len) -> needs to be (1, 1, 1, Seq_Len)?
                # No, batch['encoder_mask'] is usually (B, 1, 1, Seq_Len) or (B, 1, Seq_Len) depending on implementation.
                # In dataset.py usually it is (1, 1, Seq_Len) for padding mask. 
                # Let's trust greedy_decode handles (1, 1, 1, Seq_Len) if we pass it correctly.
                # Actually greedy_decode takes what model.encode takes.
                
                model_out = greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device)
                model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
                predicted.append(model_out_text)
                
                # Print to console just a few (2 examples)
                if count < 2:
                    print_msg('-'*console_width)
                    print_msg(f"{f'SOURCE: ':>12}{src_text[i]}")
                    print_msg(f"{f'TARGET: ':>12}{tgt_text[i]}")
                    print_msg(f"{f'PRED GREEDY: ':>12}{model_out_text}")
                    
                    # Also compute beam search for visualization
                    beam_out = beam_search_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, beam_size=3)
                    beam_out_text = tokenizer_tgt.decode(beam_out.detach().cpu().numpy())
                    print_msg(f"{f'PRED BEAM: ':>12}{beam_out_text}")
                    
                    if not model_out_text:
                        print_msg(f"{f'NOTE: ':>12}Greedy prediction is empty (likely only EOS predicted).")
                    
                    count += 1
            
    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0
    writer.add_scalar('val_loss', avg_val_loss, global_step)
    
    # Compute BLEU
    metric = get_metric("bleu", device)
    # BLEU expects list of strings for preds, list of list of strings for refs? 
    # TorchMetrics BLEUScore expects preds (List[str]) and target (List[List[str]]) usually? 
    # Or just List[str] if 1 ref.
    # Let's check torchmetrics documentation or error. Usually target is List[str] or List[List[str]].
    # For single reference, List[str] might work or List[[ref]]. 
    # Checking my knowledge: BLEUScore(preds, target). target shape same as preds? 
    # No, target is usually list of references. 
    # Let's wrap each expected in a list just in case: [[ref1], [ref2], ...] is standard for multi-ref.
    # But usually 1-1 is fine if allowed. Let's try simple list first, if fail then list of lists.
    # Actually, let's look at `metric.py` usage. It just returns `BLEUScore()`.
    # I'll update `predicted` and `expected` to be compatible.
    
    # expected needs to be list of strings? 
    # Torchmetrics BLEU: preds: Sequence[str], target: Sequence[Sequence[str]]
    
    try:
        bleu = metric(predicted, [[x] for x in expected])
        writer.add_scalar('validation_bleu', bleu, global_step)
        print_msg(f"Validation Loss: {avg_val_loss:.3f} | BLEU: {bleu:.4f}")
    except Exception as e:
        print_msg(f"Validation Loss: {avg_val_loss:.3f} | BLEU Error: {e}")
        bleu = 0.0

    writer.flush()
    return avg_val_loss, bleu, predicted, expected, source_texts

if __name__ == '__main__':
    config = get_config()
    # config.training.batch_size = 128
    # config.training.num_workers = 16
    train_model(config)
