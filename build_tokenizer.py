import os
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, Split, ByteLevel as ByteLevelPreTokenizer
from tokenizers.normalizers import NFC
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import ByteLevel as ByteLevelProcessor

SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
GPT_2_SPLIT_REGEX = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def get_or_build_tokenizer(ds, lang):
    tokenizer_path = Path(f"tokenizer_{lang}.json")

    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))
    
    def get_training_corpus():
        for item in ds:
            yield item["translation"][lang]
    
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = Sequence(
        [
            Split(pattern=GPT_2_SPLIT_REGEX, behavior="isolated"),
            ByteLevelPreTokenizer(add_prefix_space=False),
        ]
    )
    trainer = BpeTrainer(
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2, 
        show_progress=True
    )
    tokenizer.train_from_iterator(get_training_corpus(), trainer)
    tokenizer.post_processor = ByteLevelProcessor()
    tokenizer.decoder = ByteLevelDecoder()

    tokenizer.save(str(tokenizer_path))
    return tokenizer