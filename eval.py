import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import os

from torchmetrics.text import BLEUScore, CharErrorRate, WordErrorRate
from tokenizers import Tokenizer
from config import get_config
from data_loader import get_dataloaders
from model import Transformer
from dataset import BilingualDataset, causal_mask
from config import Config, get_weights_file_path, latest_weights_file_path

def get_metric(metric_name: str, device: str):
    if metric_name == "bleu":
        return BLEUScore().to(device)
    elif metric_name == "cer":
        return CharErrorRate().to(device)
    elif metric_name == "wer":
        return WordErrorRate().to(device)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

BEAM_SIZE = 3 
NUM_SAMPLES = 1000

def load_model(config, device, checkpoint_path=None):
    tokenizer_src = Tokenizer.from_file(config.dataset.tokenizer_file.format(config.dataset.src_lang))
    tokenizer_tgt = Tokenizer.from_file(config.dataset.tokenizer_file.format(config.dataset.tgt_lang))
    
    model = Transformer(
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        activation=config.model.activation,
        dropout=config.model.dropout,
        num_encoder_blocks=config.model.num_encoder_blocks,
        num_decoder_blocks=config.model.num_decoder_blocks,
        src_vocab_size=tokenizer_src.get_vocab_size(),
        tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
        max_seq_len=config.model.max_seq_len,
        tie_weights=config.model.tie_weights,
        use_rmsnorm=config.model.use_rmsnorm,
        use_qknorm=config.model.use_qknorm,
        use_rope=config.model.use_rope,
        use_swiglu=config.model.use_swiglu,
        rope_base=config.model.rope_base
    ).to(device)
    if checkpoint_path is None:
        weights_path = latest_weights_file_path(config)
    else:
        weights_path = get_weights_file_path(config, checkpoint_path)
        
    if not weights_path:
        print(f"❌ Checkpoint not found at: {checkpoint_path}")
        return None, None, None
        
    print(f"Loading: {weights_path}")
    state = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    
    if 'epoch' in state:
        print(f"   Epoch: {state['epoch']}")
    if 'val_loss' in state:
        print(f"   Validation Loss: {state['val_loss']:.4f}")
        
    model.eval()
    return model, tokenizer_src, tokenizer_tgt

def beam_search_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[BOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    candidates = [(0.0, decoder_input)]
    finished = []

    for _ in range(max_len):
        if not candidates: break
        new_candidates = []
        
        for score, seq in candidates:
            if seq[0, -1].item() == eos_idx:
                finished.append((score, seq))
                continue
            
            decoder_mask = causal_mask(seq.size(1)).type_as(source_mask).to(device)
            out = model.decode(seq, encoder_output, source_mask, decoder_mask)
            logits = model.project(out[:, -1])

            log_prob = F.log_softmax(logits, dim=1)
            
            topk_prob, topk_idx = torch.topk(log_prob, BEAM_SIZE, dim=1)
            
            for i in range(BEAM_SIZE):
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                new_score = score + topk_prob[0][i].item()
                new_seq = torch.cat([seq, token], dim=1)
                new_candidates.append((new_score, new_seq))
        
        candidates = sorted(
            new_candidates, 
            key=lambda x: x[0] / len(x[1]), 
            reverse=True
        )[:BEAM_SIZE]

        if len(finished) >= BEAM_SIZE:
            break

    final = candidates + finished
    if not final: 
        return decoder_input.squeeze(0)
    
    best = max(final, key=lambda x: x[0] / len(x[1]))
    return best[1].squeeze(0)

def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[BOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    for _ in range(max_len):
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)
        logits = model.project(out[:, -1])

        prob = F.softmax(logits, dim=1)
        next_token = torch.argmax(prob, dim=1).unsqueeze(0)

        decoder_input = torch.cat([decoder_input, next_token], dim=1)

        if next_token.item() == eos_idx:
            break

    return decoder_input.squeeze(0)

def evaluate():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, tokenizer_src, tokenizer_tgt = load_model(
        config, device
    )
    if not model: 
        return
    
    _, val_dataloader, _, _ = get_dataloaders(config)
    metric = get_metric("bleu", device)
    
    predicted = []
    expected = []
    source_texts = []
    
    print("Running Beam Search...")
    
    total_samples = 0 
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            src_texts = batch['src_texts']
            tgt_texts = batch['tgt_texts']

            for i in range(encoder_input.size(0)):
                
                if NUM_SAMPLES and total_samples >= NUM_SAMPLES:
                    break
                source = encoder_input[i].unsqueeze(0)
                source_mask = encoder_mask[i].unsqueeze(0)
                
                out = beam_search_decode(
                    model, source, source_mask, tokenizer_tgt, 
                    config.model.max_seq_len, device
                )
                
                pred_text = tokenizer_tgt.decode(out.detach().cpu().numpy())
                predicted.append(pred_text)
                expected.append([tgt_texts[i]])
                source_texts.append(src_texts[i])
                
                total_samples += 1

    score = metric(predicted, expected)
    print(f"BLEU SCORE: {score.item():.4f}")
    
    print("\nSome examples:")
    num_examples = min(5, len(predicted))
    for i in range(num_examples):
        print(f"\nExample {i+1}:")
        print(f"  Source:    {source_texts[i]}")
        print(f"  Reference: {expected[i][0]}")
        print(f"  Predicted: {predicted[i]}")

if __name__ == "__main__":
    evaluate()