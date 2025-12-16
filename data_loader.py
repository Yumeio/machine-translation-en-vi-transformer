import os
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer

from config import Config
from dataset import BilingualDataset
from collate_fn import CollateFn
from build_tokenizer import get_or_build_tokenizer

def get_dataloaders(config: Config):
    """
    Constructs and returns the training and validation DataLoaders.
    
    Args:
        config (Config): Configuration object containing Model, Training, and Dataset configs.
    
    Returns:
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
    """

    # 1. Load Datasets
    train_ds_raw = load_dataset("parquet", data_files=config.dataset.train_path, split="train")
    val_ds_raw = load_dataset("parquet", data_files=config.dataset.validation_path, split="train")

    # 2. Load Tokenizers
    tokenizer_src_path = config.dataset.tokenizer_file.format(config.dataset.src_lang)
    tokenizer_tgt_path = config.dataset.tokenizer_file.format(config.dataset.tgt_lang)
    
    if not os.path.exists(tokenizer_src_path) or not os.path.exists(tokenizer_tgt_path):
        get_or_build_tokenizer(train_ds_raw, config.dataset.src_lang)
        get_or_build_tokenizer(train_ds_raw, config.dataset.tgt_lang)

    tokenizer_src = Tokenizer.from_file(tokenizer_src_path)
    tokenizer_tgt = Tokenizer.from_file(tokenizer_tgt_path)

    # 3. Create BilingualDataset instances
    train_ds = BilingualDataset(
        dataset=train_ds_raw,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        src_lang=config.dataset.src_lang,
        tgt_lang=config.dataset.tgt_lang,
        max_seq_len=config.model.max_seq_len,
    )

    val_ds = BilingualDataset(
        dataset=val_ds_raw,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        src_lang=config.dataset.src_lang,
        tgt_lang=config.dataset.tgt_lang,
        max_seq_len=config.model.max_seq_len,
    )

    # 4. Collate Function
    # We need pad_token_id from the target tokenizer
    pad_token_id = tokenizer_tgt.token_to_id("[PAD]")
    
    if pad_token_id is None:
        raise ValueError("Tokenizer does not contain '[PAD]' token.")

    collate_fn = CollateFn(pad_token_id=pad_token_id)

    # 5. Create DataLoaders
    train_dataloader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        num_workers=config.training.num_workers,
        prefetch_factor=config.training.prefetch_factor,
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        num_workers=config.training.num_workers,
        prefetch_factor=config.training.prefetch_factor,
    )

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt