import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BiLingualData(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_trgt, lang_src, lang_trgt, seq_len):
        super().__init__()

        self.ds=ds
        self.tokenizer_src=tokenizer_src
        self.tokenizer_trgt=tokenizer_trgt
        self.lang_src=lang_src
        self.lang_trgt=lang_trgt
        self.seq_len=seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')],dtype=torch.int64)
        self.eos_token=torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token=torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair=self.ds[index]
        src_text=src_target_pair[self.lang_src]
        trgt_text=src_target_pair[self.lang_trgt]

        enc_input_tokens=self.tokenizer_src.encode(src_text).ids
        dec_input_tokens=self.tokenizer_trgt.encode(trgt_text).ids

        enc_num_padding_tokens=self.seq_len-len(enc_input_tokens)-2
        dec_num_padding_tokens=self.seq_len-len(dec_input_tokens)-1

        if enc_num_padding_tokens<0 or dec_num_padding_tokens<0:
            raise ValueError("Sentence is too long")

        enc_input=torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token.item()]*enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        dec_input=torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token.item()]*dec_num_padding_tokens, dtype=torch.int64)
            ], dim=0
        )

        label=torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token.item()]*dec_num_padding_tokens, dtype=torch.int64)
            ], dim=0
        )
        assert enc_input.size(0)==self.seq_len
        assert dec_input.size(0)==self.seq_len
        assert label.size(0)==self.seq_len

        return {
            "encoder_input":enc_input,
            "decoder_input":dec_input,
            "encoder_mask":(enc_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask":(dec_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(dec_input.size(0)),
            "label":label,
            "src_text":src_text,
            "trgt_txt":trgt_text
        }

def causal_mask(size):
    mask=torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask==0