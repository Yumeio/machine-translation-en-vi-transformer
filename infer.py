import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from pathlib import Path
from model import Transformer
from config import get_config, latest_weights_file_path, get_weights_file_path
from dataset import causal_mask
import os
import sys

BEAM_SIZE = 3

def load_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Đang sử dụng thiết bị: {device} ---")
    
    src_tok_path = config.dataset.tokenizer_file.format(config.dataset.src_lang)
    tgt_tok_path = config.dataset.tokenizer_file.format(config.dataset.tgt_lang)
    
    if not os.path.exists(src_tok_path) or not os.path.exists(tgt_tok_path):
        print(f"LỖI: Không tìm thấy file tokenizer tại {src_tok_path} hoặc {tgt_tok_path}")
        sys.exit(1)

    tokenizer_src = Tokenizer.from_file(src_tok_path)
    tokenizer_tgt = Tokenizer.from_file(tgt_tok_path)
    
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
        max_seq_len=config.model.max_seq_len
    ).to(device)
    
    weights_path = get_weights_file_path(config, "29")
    if weights_path:
        print(f"--- Đang load weights từ: {weights_path} ---")
        try:
            state = torch.load(weights_path, map_location=device)
            model.load_state_dict(state['model_state_dict'])
            print("--- Load weights THÀNH CÔNG ---")
        except Exception as e:
            print(f"LỖI khi load weights: {e}")
            sys.exit(1)
    else:
        print("!!! CẢNH BÁO: Không tìm thấy file weights nào. Model sẽ dùng random weights (dịch ra rác) !!!")
        
    model.eval()
    return model, tokenizer_src, tokenizer_tgt, device

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[BOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

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
        
        candidates = sorted(new_candidates, key=lambda x: x[0], reverse=True)[:BEAM_SIZE]

    final = candidates + finished
    if not final: return decoder_input.squeeze(0)
    best = max(final, key=lambda x: x[0] / len(x[1]))
    
    return best[1].squeeze(0)

def run_translation():
    text = "My family was not poor , and myself , I had never experienced hunger ."
    print(f"Câu gốc: {text}")
    
    config = get_config()
    model, tokenizer_src, tokenizer_tgt, device = load_model(config)
    
    try:
        enc_input_tokens = tokenizer_src.encode(text).ids
        source = torch.tensor(enc_input_tokens).view(1, -1).to(device)
        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(1).unsqueeze(2).int().to(device)
        
        print("--- Bắt đầu dịch (Greedy) ---")
        greedy_out = greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, config.model.max_seq_len, device)
        print(f"Kết quả Greedy: {tokenizer_tgt.decode(greedy_out.detach().cpu().numpy())}")

        print("\n--- Bắt đầu dịch (Beam Search) ---")
        beam_out = beam_search_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, config.model.max_seq_len, device, beam_size=3)
        print(f"Kết quả Beam: {tokenizer_tgt.decode(beam_out.detach().cpu().numpy())}")
        
    except Exception as e:
        print(f"\n!!! LỖI TRONG QUÁ TRÌNH DỊCH: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_translation()