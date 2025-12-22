import os
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer

from build_tokenizer import get_or_build_tokenizer
from dataset import BilingualDataset
from collate_fn import CollateFn
from config import Config

def get_dataloaders(config: Config):
    print("Loading datasets...")
    train_ds_raw = load_dataset("parquet", data_files=config.dataset.train_path, split="train")
    val_ds_raw = load_dataset("parquet", data_files=config.dataset.validation_path, split="train")
    print(f"✅ Train samples: {len(train_ds_raw):,}")
    print(f"✅ Validation samples: {len(val_ds_raw):,}")
    
    tokenizer_src_path = config.dataset.tokenizer_file.format(config.dataset.src_lang)
    tokenizer_tgt_path = config.dataset.tokenizer_file.format(config.dataset.tgt_lang)
    
    if not os.path.exists(tokenizer_src_path) or not os.path.exists(tokenizer_tgt_path):
        print("\nBuilding tokenizers...")
        get_or_build_tokenizer(train_ds_raw, config.dataset.src_lang)
        get_or_build_tokenizer(train_ds_raw, config.dataset.tgt_lang)
        
    print("\nLoading tokenizers...")
    tokenizer_src = Tokenizer.from_file(tokenizer_src_path)
    tokenizer_tgt = Tokenizer.from_file(tokenizer_tgt_path)
    
    print("\nCreating Bilingual dataset...")
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
    
    pad_token_id_src = tokenizer_src.token_to_id("[PAD]")
    pad_token_id_tgt = tokenizer_tgt.token_to_id("[PAD]")
    
    if pad_token_id_src is None:
        raise ValueError("Source tokenizer does not contain '[PAD]' token.")
    if pad_token_id_tgt is None:
        raise ValueError("Target tokenizer does not contain '[PAD]' token.")
    
    collate_fn = CollateFn(
        pad_token_id_src=pad_token_id_src,
        pad_token_id_tgt=pad_token_id_tgt
    )
    
    use_persistent_workers = config.training.num_workers > 0
    print("\nCreating DataLoaders...")
    train_dataloader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=use_persistent_workers,
        num_workers=config.training.num_workers,
        prefetch_factor=config.training.prefetch_factor if config.training.num_workers > 0 else None,
    )
    
    val_dataloader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=use_persistent_workers,
        num_workers=config.training.num_workers,
        prefetch_factor=config.training.prefetch_factor if config.training.num_workers > 0 else None,
    )

    print(f"✅ Train batches: {len(train_dataloader):,}")
    print(f"✅ Validation batches: {len(val_dataloader):,}")

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt