'''GPT-1 architecture'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Define key parameters
batch_size = 64
d_model = 512
nhead = 8
num_decoder_layers = 8
dim_feedforward = 2048
dropout = 0.1
vocab_size = 30000
max_len = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=max_len):
        super().__init__()
        self.d_model = d_model
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Define positions
        positions = torch.arange(0, max_len).unsqueeze(0)
        self.register_buffer('positions', positions)
    
    def forward(self, x):
        B, T, C = x.shape
        # Get positions for the actual sequence length
        pos = self.pos_emb(self.positions[:, :T])
        # Expand positions to match batch size
        pos = pos.expand(B, -1, -1)
        # Add positional encoding to input
        return x + pos

# Define single Attention Head
class AttentionHead(nn.Module):
    def __init__(self, head_size, masking=True):
        super(AttentionHead, self).__init__()
        self.Q = nn.Linear(d_model, head_size) # Query: What the token wants
        self.K = nn.Linear(d_model, head_size) # Key: What the token has
        self.V = nn.Linear(d_model, head_size) # Value: What the token can communicate if other tokens are interested in it
        self.d_model = d_model
        self.softmax = nn.Softmax(dim=-1)
        self.masking = masking
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x, padding_mask=None): 
        B, T, C = x.shape
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        attention_weights = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)

        if padding_mask is not None:
            attention_weights = attention_weights.masked_fill(padding_mask.unsqueeze(1), float('-inf'))

        if self.masking: # Causal masking for decoder self-attention
            tril = torch.tril(torch.ones(T, T)).to(device)
            attention_weights = attention_weights.masked_fill(tril == 0, float('-inf'))

        attention_weights = self.softmax(attention_weights)
        attention_weights = self.dropout(attention_weights)
        return attention_weights @ v

# Define Multi-Head Attention
class MultiHeadAttention(nn.Module):
    '''Tokens look at each other via Multi-Head Attention'''
    def __init__(self, num_heads, masking=True):
        super(MultiHeadAttention, self).__init__()
        self.head_size = d_model // num_heads
        assert self.head_size * num_heads == d_model, "d_model must be divisible by num_heads"
        self.heads = nn.ModuleList([
            AttentionHead(self.head_size, masking) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, padding_mask=None):
        out = torch.cat([head(x, padding_mask) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# Define Feedforward Layer
class FeedForward(nn.Module):
    '''Tokens communicate/talk with each other via FeedForward Layer'''
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.gelu(self.linear1(x))
        x = self.dropout(self.linear2(x))
        return x
    
# Define Decoder Layer
class Decoder(nn.Module):
    '''MHA(self,masked) + x -> LN ->  FF + x -> LN'''
    def __init__(self, d_model, nhead, dropout):
        super(Decoder, self).__init__()
        self.self_attn = MultiHeadAttention(nhead)
        self.ff = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, decoder_padding_mask=None):
        self_attn_out = self.dropout(self.self_attn(x, decoder_padding_mask))
        x = self.norm1(x + self_attn_out)
        ff_out = self.dropout(self.ff(x))
        x = self.norm2(x + ff_out)
        return x

# Define GPT-1 model
class GPT_1(nn.Module):
    '''GPT-1 model'''
    def __init__(self, vocab_size, d_model, nhead, num_decoder_layers, dropout, max_len=max_len):
        super(GPT_1, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.decoder_layers = nn.ModuleList([
            Decoder(d_model, nhead, dropout) for _ in range(num_decoder_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, decoder_padding_mask=None):
        x = self.token_emb(x)
        x = self.pos_emb(x)
        for layer in self.decoder_layers:
            x = layer(x, decoder_padding_mask)
        x = self.fc(x)
        return x
    
    def generate(self, max_len=128, batch_size=64):
        '''Generate text'''
        decoder_input = torch.ones((batch_size,1), dtype=torch.long, device=device) * 5 # [BOS] token

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits = self(decoder_input)

            # Get the last predicted token
            logits = logits[:, -1, :].squeeze(1)
            next_token = torch.argmax(logits, dim=-1).unsqueeze(1)

            # Force finished sequences to keep producing [EOS]
            next_token = torch.where(finished.unsqueeze(-1), torch.tensor(6, device=device), next_token)

            # Append to decoder input
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            # Update finished status
            finished = finished | (next_token.squeeze(-1) == 6)

            # If all sequences are finished, break
            if finished.all():
                break
        
        return decoder_input