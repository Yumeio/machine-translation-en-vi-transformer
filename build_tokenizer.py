import os 
from pathlib import Path
from tokenizers import (
    Tokenizer, 
    models, 
    pre_tokenizers,
    decoders, 
    trainers, 
    processors
)
from tokenizers.implementations import ByteLevelBPETokenizer

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]
        
def get_or_build_tokenizer(ds, lang):
    tokenizer_path = Path(f"tokenizer_{lang}.json")
    if not Path.exists(tokenizer_path):
        print(f"Tokenizer not found. Building tokenizer for language: {lang}...")

        special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]

        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        
        trainer = trainers.BpeTrainer(
            special_tokens=special_tokens,
            show_progress=True,
            min_frequency=2
        )
        
        print(f"Training tokenizer on {lang} data...")
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                ("[BOS]", tokenizer.token_to_id("[BOS]")),
                ("[EOS]", tokenizer.token_to_id("[EOS]")),
            ]
        )
        
        # Save tokenizer
        tokenizer.save(str(tokenizer_path))
        print(f"✅ Tokenizer saved at: {tokenizer_path}")
        
        # Print vocab size
        vocab_size = tokenizer.get_vocab_size()
        print(f"   Vocabulary size: {vocab_size:,}")
    else:
        print(f"✅ Found existing tokenizer at: {tokenizer_path}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        vocab_size = tokenizer.get_vocab_size()
        print(f"   Vocabulary size: {vocab_size:,}")
        
    for token in ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]:
        token_id = tokenizer.token_to_id(token)
        print(f"   Token: {token} | ID: {token_id}")
        if token_id is None:
            raise ValueError(f"Special token {token} not found in tokenizer!")
        
    return tokenizer