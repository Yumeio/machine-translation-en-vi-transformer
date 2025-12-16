import torch
from tokenizers import Tokenizer
from pathlib import Path
from model import Transformer
from config import get_config, latest_weights_file_path
from dataset import causal_mask

def load_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        max_seq_len=config.model.max_seq_len
    ).to(device)
    
    weights_path = latest_weights_file_path(config)
    if weights_path:
        print(f"Loading weights from {weights_path}")
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
    else:
        print("No weights found. Using random weights.")
        
    model.eval()
    return model, tokenizer_src, tokenizer_tgt, device

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_token_id = tokenizer_tgt.token_to_id('[BOS]')
    eos_token_id = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_token_id).type_as(source).to(device)
    
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_token_id:
            break

    return decoder_input.squeeze(0)

def beam_search_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, beam_size=3):
    sos_token_id = tokenizer_tgt.token_to_id('[BOS]')
    eos_token_id = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_token_id).type_as(source).to(device)
    
    # (score, sequence)
    candidates = [(0.0, decoder_input)]
    
    while True:
        if all([cand[1].size(1) == max_len for cand in candidates]):
            break
            
        new_candidates = []
        
        for score, seq in candidates:
            if seq.size(1) == max_len or seq[0, -1].item() == eos_token_id:
                new_candidates.append((score, seq))
                continue
                
            decoder_mask = causal_mask(seq.size(1)).type_as(source_mask).to(device)
            out = model.decode(seq, encoder_output, source_mask, decoder_mask)
            prob = model.project(out[:, -1])
            prob = torch.nn.functional.log_softmax(prob, dim=1) # Log softmax for scores
            
            # Get top k
            topk_prob, topk_idx = torch.topk(prob, beam_size, dim=1)
            
            for i in range(beam_size):
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                new_score = score + topk_prob[0][i].item()
                new_seq = torch.cat([seq, token], dim=1)
                new_candidates.append((new_score, new_seq))
                
        # Sort and keep top k
        candidates = sorted(new_candidates, key=lambda x: x[0], reverse=True)[:beam_size]
        
        if all([cand[1][0, -1].item() == eos_token_id for cand in candidates]):
            break
            
    # Return the best sequence
    return candidates[0][1].squeeze(0)

def translate_sentence(model, tokenizer_src, tokenizer_tgt, device, text, beam_size=None, max_seq_len=100):
    # Encode source text
    enc_input_tokens = tokenizer_src.encode(text).ids
    source = torch.tensor(enc_input_tokens).view(1, -1).to(device)
    source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(1).unsqueeze(2).int().to(device)
    
    # Decode
    if beam_size:
        model_out = beam_search_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_seq_len, device, beam_size=beam_size)
    else:
        model_out = greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_seq_len, device)
    
    # Detokenize
    translated_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
    return translated_text

def translate(text, beam_size=None):
    config = get_config()
    model, tokenizer_src, tokenizer_tgt, device = load_model(config)
    return translate_sentence(model, tokenizer_src, tokenizer_tgt, device, text, beam_size, config.model.max_seq_len)

if __name__ == "__main__":
    t = "One time , our bus was stopped and boarded by a Chinese police officer ."
    print(f"Source: {t}")
    
    config = get_config()
    model, tokenizer_src, tokenizer_tgt, device = load_model(config)
    
    # Greedy
    greedy_out = translate_sentence(model, tokenizer_src, tokenizer_tgt, device, t, beam_size=None, max_seq_len=config.model.max_seq_len)
    print(f"Greedy: {greedy_out}")
    
    # Beam Search
    beam_out = translate_sentence(model, tokenizer_src, tokenizer_tgt, device, t, beam_size=4, max_seq_len=config.model.max_seq_len)
    print(f"Beam (k=4): {beam_out}")
