import os
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence
from tokenizers.processors import TemplateProcessing

SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]

def get_or_build_tokenizer(ds, lang):
    tokenizer_path = Path(f"tokenizer_{lang}.json")

    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))
    
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2, 
        show_progress=True
    )

    def get_training_corpus():
        for item in ds:
            yield item["translation"][lang]

    tokenizer.train_from_iterator(get_training_corpus(), trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )

    tokenizer.save(str(tokenizer_path))
    return tokenizer