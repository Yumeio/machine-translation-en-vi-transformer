from torch.nn.utils.rnn import pad_sequence
from dataset import causal_mask

class CollateFn:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        encoder_inputs = [item['encoder_input'] for item in batch]
        decoder_inputs = [item['decoder_input'] for item in batch]
        decoder_outputs = [item['decoder_output'] for item in batch]
        src_texts = [item['src_text'] for item in batch]
        tgt_texts = [item['tgt_text'] for item in batch]
        
        encoder_input_batch = pad_sequence(
            encoder_inputs,
            batch_first=True,
            padding_value=self.pad_token_id
        )
        decoder_input_batch = pad_sequence(
            decoder_inputs,
            batch_first=True,
            padding_value=self.pad_token_id
        )
        decoder_output_batch = pad_sequence(
            decoder_outputs,
            batch_first=True,
            padding_value=self.pad_token_id
        )

        encoder_mask = (encoder_input_batch != self.pad_token_id).unsqueeze(1).unsqueeze(2).int()
        decoder_seq_len = decoder_input_batch.size(1)
        decoder_mask = (decoder_input_batch != self.pad_token_id).unsqueeze(1).unsqueeze(2).int() & causal_mask(decoder_seq_len)
        
        return {
            'encoder_input': encoder_input_batch,
            'decoder_input': decoder_input_batch,
            'decoder_output': decoder_output_batch,
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask,
            'src_texts': src_texts,
            'tgt_texts': tgt_texts
        }