# import os
# import sys
import math
# import random
# import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")  # In case of MacOS with Metal
    return torch.device("cpu")


device = get_device()
torch.manual_seed(21339)  # lock in seed for reproducibility, delete later
# random.seed(21339)

# Provided hyperparameters from assigment sheet
experiment_hyperparameters = {
    "1": {
        "EMB_DIM": 128,
        "N_LAYERS": 1,
        "N_HEADS": 8,
        "FORWARD_DIM": 512,
        "DROPOUT": 0.05,
        "LEARNING_RATE": 7e-4,
        "BATCH_SIZE": 64,
        "GRAD_CLIP": 1,
        "OPTIMIZER": "AdamW",
    },
    "2": {
        "EMB_DIM": 128,
        "N_LAYERS": 2,
        "N_HEADS": 8,
        "FORWARD_DIM": 256,
        "DROPOUT": 0.15,
        "LEARNING_RATE": 2e-4,
        "BATCH_SIZE": 16,
        "GRAD_CLIP": 1,
        "OPTIMIZER": "AdamW",
    },
    "3": {
        "EMB_DIM": 128,
        "N_LAYERS": 2,
        "N_HEADS": 8,
        "FORWARD_DIM": 256,
        "DROPOUT": 0.15,
        "LEARNING_RATE": 2e-4,
        "BATCH_SIZE": 16,
        "GRAD_CLIP": 1,
        "OPTIMIZER": "AdamW",
    },
}


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.query_linear = nn.Linear(emb_dim, num_heads * self.head_dim, bias=False) # W_Q
        self.key_linear = nn.Linear(emb_dim, num_heads * self.head_dim, bias=False) # W_K
        self.value_linear = nn.Linear(emb_dim, num_heads * self.head_dim, bias=False) # W_V
        self.out_linear = nn.Linear(num_heads * self.head_dim, emb_dim, bias=False)


    def forward(self, query, key, value, mask=None):
        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)
        batch_size, seq_len, _ = Q.shape
        Q = Q.view(Q.shape[0], Q.shape[1], self.num_heads, self.head_dim)
        K = K.view(K.shape[0], K.shape[1], self.num_heads, self.head_dim)
        V = V.view(V.shape[0], V.shape[1], self.num_heads, self.head_dim)
        # perform Q * K^T
        key_out = torch.einsum("bqhd,bkhd->bhqk", Q, K) / math.sqrt(self.head_dim)
        if mask is not None:
            key_out = key_out.masked_fill(mask == 0, -1e20)

        attention = torch.softmax(
            key_out, dim=-1
        )

        out = torch.einsum("bhqk,bkhd->bqhd", attention, V)
        out = out.contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)

        out = self.out_linear(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, forward_dim):
        super().__init__()
        self.attention = MultiHeadAttention(emb_dim, num_heads)
        self.norm1 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, emb_dim)
        )

    def forward(self, query, key, value, mask):
        attention_out = self.attention(query, key, value, mask)
        attention_out = self.norm1(self.drop(attention_out + query))
        ffn_out = self.ffn(attention_out)
        ffn_out = self.norm2(self.drop(ffn_out + attention_out))
        
        return ffn_out


def get_sinusoid_table(max_len, emb_dim):
    def get_angle(pos, i, emb_dim):
        return pos / 10000 ** ((2 * (i // 2)) / emb_dim)

    sinusoid_table = torch.zeros(max_len, emb_dim)
    for pos in range(max_len):
        for i in range(emb_dim):
            if i % 2 == 0:
                sinusoid_table[pos, i] = math.sin(get_angle(pos, i, emb_dim))
            else:
                sinusoid_table[pos, i] = math.cos(get_angle(pos, i, emb_dim))
    return sinusoid_table


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        num_layers,
        num_heads,
        forward_dim,
        dropout,
        max_len,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_table(max_len, emb_dim), freeze=True)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([TransformerBlock(emb_dim, num_heads, dropout, forward_dim) for _ in range(num_layers)])

    def forward(self, x, mask):
        pos_encodings = self.pos_emb(torch.arange(start=1, end=x.size(1)+1, device=x.device))
        x = self.drop(self.emb(x) + pos_encodings)
        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, forward_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim, eps=1e-6)
        self.attention = MultiHeadAttention(emb_dim, num_heads)
        self.transformer_block = TransformerBlock(emb_dim, num_heads, dropout, forward_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, tgt_mask):
        self_attention = self.attention(x, x, x, tgt_mask)
        self_attention = self.norm(self.drop(self_attention + x)) # query
        out = self.transformer_block(self_attention, key, value, src_mask)
        return  out

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        num_layers,
        num_heads,
        forward_dim,
        dropout,
        max_len
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.drop = nn.Dropout(dropout)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.layers = nn.ModuleList([DecoderBlock(emb_dim, num_heads, forward_dim, dropout) for _ in range(num_layers)])
        self.out = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        embeddings = self.emb(x)
        pos_encodings = self.pos_emb(torch.arange(start=0, end=x.size(1), device=x.device)) # no shifting
        x = self.drop(embeddings + pos_encodings)
        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, src_mask, tgt_mask)
        return self.out(x)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        emb_dim=512,
        num_layers=6,
        num_heads=8,
        forward_dim=2048,
        dropout=0.0,
        max_len=128,
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, emb_dim, num_layers, num_heads, forward_dim, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, emb_dim, num_layers, num_heads, forward_dim, dropout, max_len)
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def create_src_mask(self, src):
        device = src.device
        # (batch_size, 1, 1, src_seq_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(device)

    def create_tgt_mask(self, tgt):
        device = tgt.device
        batch_size, tgt_len = tgt.shape
        tgt_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask * torch.tril(torch.ones((tgt_len, tgt_len))).expand(
            batch_size, 1, tgt_len, tgt_len
        ).to(device)
        return tgt_mask

    def forward(self, src, tgt):
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)
        encoder_out = self.encoder(src, src_mask)
        return self.decoder(tgt, encoder_out, src_mask, tgt_mask)


model = Transformer(
    src_vocab_size=200,
    tgt_vocab_size=220,
    src_pad_idx=0,
    tgt_pad_idx=0,
).to(device)

# source input: batch size 4, sequence length of 75
src_in = torch.randint(0, 200, (4, 75)).to(device)

# target input: batch size 4, sequence length of 80
tgt_in = torch.randint(0, 220, (4, 80)).to(device)

# expected output shape of the model
expected_out_shape = torch.Size([4, 80, 220])

with torch.no_grad():
    out = model(src_in, tgt_in)
#    print("Output shape:", out.shape)  # Debug step, left in for convenience

assert (
    out.shape == expected_out_shape
), f"wrong output shape, expected: {expected_out_shape}"
