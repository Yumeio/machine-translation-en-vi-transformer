import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer_src,
        tokenizer_tgt,
        src_lang,
        tgt_lang,
        max_seq_len,
    ):
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_seq_len = max_seq_len

        self.sos_token_id = tokenizer_tgt.token_to_id("[BOS]")
        self.eos_token_id = tokenizer_tgt.token_to_id("[EOS]")
        self.pad_token_id = tokenizer_tgt.token_to_id("[PAD]")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            src_target_pair = self.dataset[idx]
            src_text = src_target_pair['translation'][self.src_lang]
            tgt_text = src_target_pair['translation'][self.tgt_lang]

            enc_input_tokens = self.tokenizer_src.encode(src_text).ids
            dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

            # Encoder: only needs EOS -> max = seq_len - 1
            if len(enc_input_tokens) > self.max_seq_len - 1:
                enc_input_tokens = enc_input_tokens[:self.max_seq_len - 1]
            
            # Decoder: needs SOS -> max = seq_len - 1
            if len(dec_input_tokens) > self.max_seq_len - 1:
                dec_input_tokens = dec_input_tokens[:self.max_seq_len - 1]

            # Encoder = tokens + [EOS] (NO SOS token)
            # Encoders don't use SOS token in standard Transformer architecture
            encoder_input = torch.cat([
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                torch.tensor([self.eos_token_id], dtype=torch.int64)
            ], dim=0)

            # Decoder input = [SOS] + tokens
            decoder_input = torch.cat([
                torch.tensor([self.sos_token_id], dtype=torch.int64),
                torch.tensor(dec_input_tokens, dtype=torch.int64)
            ], dim=0)

            # Decoder output = tokens + [EOS]
            decoder_output = torch.cat([
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.eos_token_id], dtype=torch.int64)
            ], dim=0)

            assert len(encoder_input) <= self.max_seq_len, \
                f"Encoder input length {len(encoder_input)} exceeds max_seq_len {self.max_seq_len}"
            assert len(decoder_input) <= self.max_seq_len, \
                f"Decoder input length {len(decoder_input)} exceeds max_seq_len {self.max_seq_len}"
            assert len(decoder_output) <= self.max_seq_len, \
                f"Decoder output length {len(decoder_output)} exceeds max_seq_len {self.max_seq_len}"

            return {
                "src_text": src_text,
                "tgt_text": tgt_text,
                "encoder_input": encoder_input,
                "decoder_input": decoder_input,
                "decoder_output": decoder_output
            }
        except Exception as e:
            print(f"Error processing sample at index {idx}: {e}")
            print(f"Source text: {src_text if 'src_text' in locals() else 'N/A'}")
            print(f"Target text: {tgt_text if 'tgt_text' in locals() else 'N/A'}")
            raise
         
def causal_mask(size):
    """causal mask for decoder self-attention"""
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0