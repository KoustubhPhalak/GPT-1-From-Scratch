import torch
from torch.utils.data import DataLoader, Dataset
from GPT_Decoder import max_len
import torch.nn.functional as F

# Create PyTorch Dataset
class WikiTextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.bos_id = tokenizer.token_to_id('[BOS]')
        self.eos_id = tokenizer.token_to_id('[EOS]')

        # Preprocess all data upfront
        self.processed_data = []
        for item in data:
            text = item['text']
            encoding = tokenizer.encode(text).ids
            seq = [self.bos_id] + encoding + [self.eos_id]
            seq = seq[:max_len]
            self.processed_data.append(torch.LongTensor(seq))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.processed_data[idx]
        
        # Create decoder input/output pairs. Right shifting is needed to make sure the model predicts the next token
        decoder_input = seq[:-1]
        decoder_target = seq[1:]

        if len(decoder_input) > self.max_len:
            encoding = encoding[:self.max_len]
        if len(decoder_target) > self.max_len:
            decoder_target = decoder_target[:self.max_len]

        return {
            'decoder_input': torch.LongTensor(decoder_input),
            'decoder_target': torch.LongTensor(decoder_target),
        }
        
def collate_fn(batch):
    pad_token = 1 # [PAD] token
    batched = {
        'decoder_input': [],
        'decoder_target': [],
        'decoder_padding_mask': []
    }

    for item in batch:
        # Pad decoder input and target
        dec_pad = max_len - len(item['decoder_input'])
        padded_dec_input = F.pad(item['decoder_input'], (0, dec_pad), value=pad_token)
        padded_dec_target = F.pad(item['decoder_target'], (0, dec_pad), value=pad_token)

        # Create boolean masks for padding (False=real token, True=padding)
        if dec_pad <= 0:
            dec_mask = torch.zeros_like(item['decoder_input'], dtype=torch.bool)
        else:
            dec_mask = torch.cat([
                torch.zeros_like(item['decoder_input'], dtype=torch.bool),
                torch.ones(dec_pad, dtype=torch.bool)
            ])
        
        batched['decoder_input'].append(padded_dec_input)
        batched['decoder_target'].append(padded_dec_target)
        batched['decoder_padding_mask'].append(dec_mask)
    
    return {k: torch.stack(v) for k, v in batched.items()}
