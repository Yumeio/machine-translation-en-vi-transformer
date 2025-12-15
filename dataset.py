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
        src_target_pair = self.dataset[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Encode the source and target texts
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # 2. Truncation (Vẫn cần cắt để tránh OOM với câu siêu dài)
        # Encoder: cần thêm SOS, EOS -> max = seq_len - 2
        if len(enc_input_tokens) > self.max_seq_len - 2:
            enc_input_tokens = enc_input_tokens[:self.max_seq_len - 2]
        
        # Decoder: cần thêm SOS -> max = seq_len - 1
        if len(dec_input_tokens) > self.max_seq_len - 1:
            dec_input_tokens = dec_input_tokens[:self.max_seq_len - 1]

        # Encoder = [SOS] + tokens + [EOS]
        encoder_input = torch.cat([
            torch.tensor([self.sos_token_id], dtype=torch.int64),
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            torch.tensor([self.eos_token_id], dtype=torch.int64)
        ], dim=0)

        # Decoder = [SOS] + tokens
        decoder_input = torch.cat([
            torch.tensor([self.sos_token_id], dtype=torch.int64),
            torch.tensor(dec_input_tokens, dtype=torch.int64)
        ], dim=0)

        # Decoder output = tokens + [EOS]
        decoder_output = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.eos_token_id], dtype=torch.int64)
        ], dim=0)

        return {
            "src_text": src_text,
            "tgt_text": tgt_text,
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "decoder_output": decoder_output
        }
         
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
