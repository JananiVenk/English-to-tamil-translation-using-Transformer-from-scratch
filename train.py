import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_sentences(ds, lang):
    for trans in ds:
        yield ds[lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path=Path(config['tokenizer_file']).format("ta")
    if not Path.exists(tokenizer_path):
        tokenzier=Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenzier.pre_tokenizer=Whitespace()
        trainer=WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"], min_frequency=2)
        tokenzier.train_from_iterator(get_all_sentences(ds, lang),trainer=trainer)
        tokenzier.save(str(tokenizer_path))
    else:
        tokenizer=Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    ds_raw=load_dataset("gopi30/english-tamil", name=f"{config['src_lang']}-{config['trgt_lang']}", split="train")
    tokenizer_src=get_or_build_tokenizer(config, ds_raw,config['src_lang'])
    tokenizer_trgt=get_or_build_tokenizer(config, ds_raw,config['trgt_lang'])

    train_ds_size=int(0.9*len(ds_raw))
    val_ds_size=len(ds_raw)-train_ds_size

    train_ds_raw, val_ds_raw=random_split(ds_raw, [train_ds_size, val_ds_size])

