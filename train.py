import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
from pathlib import Path
import warnings
import matplotlib.pyplot as plt

from config import get_config, save_config
from data_loader import get_dataloaders
from model import Transformer
from lr_scheduler import LRScheduler
from optim import AdamW

os.environ["TOKENIZER_PARALLELISM"] = "False"

def save_loss_plot(train_losses, val_losses, filename="loss_plot.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='x')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Đã lưu biểu đồ Loss tại: {filename}")

def get_model(config, vocab_src_len, vocab_tgt_len, device):
    model = Transformer(
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        activation=config.model.activation,
        dropout=config.model.dropout,
        num_encoder_blocks=config.model.num_encoder_blocks,
        num_decoder_blocks=config.model.num_decoder_blocks,
        src_vocab_size=vocab_src_len,
        tgt_vocab_size=vocab_tgt_len,
        max_seq_len=config.model.max_seq_len,
        tie_weights=config.model.tie_weights,
        use_rmsnorm=config.model.use_rmsnorm,
        use_qknorm=config.model.use_qknorm,
        use_rope=config.model.use_rope,
        use_swiglu=config.model.use_swiglu,
        rope_base=config.model.rope_base
    )
    
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)
    
    return model

def run_validation_loss(model, val_dataloader, loss_fn, device, use_amp=False):
    model.eval()
    total_val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['decoder_output'].to(device)

            with autocast(enabled=use_amp):
                model_to_use = model.module if isinstance(model, nn.DataParallel) else model
                encoder_output = model_to_use.encode(encoder_input, encoder_mask)
                decoder_output = model_to_use.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
                proj_output = model_to_use.project(decoder_output)

                loss = loss_fn(proj_output.view(-1, proj_output.size(-1)), label.view(-1))

            total_val_loss += loss.item()
            num_batches += 1
            
    return total_val_loss / num_batches if num_batches > 0 else 0

def train_model():
    config = get_config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
        
        # Check BF16 support (T4 doesn't have it)
        bf16_support = torch.cuda.is_bf16_supported()
        print(f"\nBF16 Support: {'✅ Yes' if bf16_support else '❌ No (using FP16 instead)'}")

    Path(config.training.model_folder).mkdir(parents=True, exist_ok=True)

    # Get dataloaders and model
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataloaders(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), device)

    # Optimizer, Loss function and LR scheduler
    optimizer = AdamW(
        model.parameters(), 
        lr=config.training.lr, 
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config.training.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id('[PAD]'), 
        label_smoothing=config.training.label_smoothing
    ).to(device)
    lr_scheduler = LRScheduler(
        optimizer=optimizer,
        d_model=config.model.d_model,
        warmup_steps=config.training.warmup_steps
    )

    # TensorBoard Writer
    writer = SummaryWriter(config.training.experiment_name)
    
    # AMP Grad Scaler
    scaler = GradScaler(enabled=config.training.use_amp)

    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    global_step = 0
    
    for epoch in range(config.training.num_epochs):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.training.num_epochs}")
        
        total_train_loss = 0
        num_train_batches = 0

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['decoder_output'].to(device)

            with autocast(enabled=config.training.use_amp):
                model_to_use = model.module if isinstance(model, nn.DataParallel) else model
                encoder_output = model_to_use.encode(encoder_input, encoder_mask)
                decoder_output = model_to_use.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
                proj_output = model_to_use.project(decoder_output)

                loss = loss_fn(
                    proj_output.view(-1, tokenizer_tgt.get_vocab_size()), 
                    label.view(-1)
                )
                
            current_lr = optimizer.param_groups[0]['lr']
            batch_iterator.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.2e}"
            })
            
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/learning_rate', current_lr, global_step)
            total_train_loss += loss.item()
            num_train_batches += 1
        
            # Backward with Gradient Scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.grad_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            global_step += 1

        avg_train_loss = total_train_loss / num_train_batches
        print(f"\n📊 Running validation for epoch {epoch+1}...")
        avg_val_loss = run_validation_loss(
            model, val_dataloader, loss_fn, device, 
            use_amp=config.training.use_amp
        )
        
        # Save loss history
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        
        # Logging
        print(f"Epoch {epoch+1}/{config.training.num_epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")  
        
        writer.add_scalar('epoch/train_loss', avg_train_loss, epoch+1)
        writer.add_scalar('epoch/val_loss', avg_val_loss, epoch+1)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            best_model_path = str(Path(config.training.model_folder) / "best_model.pt")
            
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'global_step': global_step,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, best_model_path)
            print(f"💾 Saved best model to {best_model_path} with val_loss={avg_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"⚠️  Val loss did not improve. Patience: {patience_counter}/{patience}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % config.training.save_steps == 0:
            model_filename = f"{config.training.model_basename}{epoch+1:03d}.pt"
            save_path = str(Path(config.training.model_folder) / model_filename)
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'global_step': global_step,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }, save_path)
            print(f"💾 Saved checkpoint to {save_path}")

        # Early stopping
        if patience_counter >= patience:
            print(f"⏸️  Early stopping triggered after {epoch+1} epochs.")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
        
    save_loss_plot(train_loss_history, val_loss_history)
    save_config(config)
    
    print("🏁 Training complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total epochs trained: {epoch+1}")
    print(f"   Total steps: {global_step}")

    
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    train_model()