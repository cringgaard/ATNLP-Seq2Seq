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
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.Q_layer = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.K_layer = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.V_layer = nn.Linear(emb_dim, num_heads * self.head_dim)
        self.output_layer = nn.Linear(num_heads * self.head_dim, emb_dim)

    def forward(self, query, key, value, mask=None):
        Q = self.Q_layer(query)
        K = self.K_layer(key)
        V = self.V_layer(value)
        batch_size, seq_len, _ = Q.shape
        Q = Q.view(Q.shape[0], Q.shape[1], self.num_heads, self.head_dim)
        K = K.view(K.shape[0], K.shape[1], self.num_heads, self.head_dim)
        V = V.view(V.shape[0], V.shape[1], self.num_heads, self.head_dim)

        key_out = torch.einsum("bqhd,bkhd->bhqk", Q, K) / math.sqrt(self.head_dim)
        if mask is not None:
            key_out = key_out.masked_fill(mask == 0, -1e20)

        attention = torch.softmax(
            key_out, dim=-1
        )  # is batch_size, num_heads, q_seq_len, k_seq_len

        out = torch.einsum("bhqk,bkhd->bqhd", attention, V)
        # should hopefully be (batch_size, len, num_heads, head_dim)
        out = out.contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)

        out = self.output_layer(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, forward_dim, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(emb_dim, num_heads)
        self.dropout = dropout  # used in forward
        self.norm1 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, emb_dim),
        )

    def forward(self, query, key, value, mask):
        attention_out = self.attention(query, key, value, mask)
        x = attention_out + query
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        x = ffn_out + x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.norm2(x)
        return x


def get_sinusoid_table(max_len, emb_dim):
    """
    Generates a sinusoid table for positional encoding, code provided as is by assignment
    """

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
        self, vocab_size, emb_dim, num_layers, num_heads, forward_dim, dropout, max_len
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.max_len = (
            max_len + 1
        )  # +1 to account for padding token (I think this is correct, but the notes are unclear to me)
        self.pos_embedding = get_sinusoid_table(self.max_len, emb_dim).to(device)
        self.dropout_layer = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(emb_dim, num_heads, forward_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask):
        x = x + 1  # shift by 1 to avoid padding token
        x = self.embedding(x)
        positions = torch.arange(x.shape[1]).expand(x.shape[0], x.shape[1]).to(device)
        x = x + self.pos_embedding[positions]
        x = self.dropout_layer(x)
        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, forward_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim, eps=1e-6)
        self.forward_dim = forward_dim
        self.dropout = dropout
        self.attention = MultiHeadAttention(emb_dim, num_heads)
        self.block = TransformerBlock(emb_dim, num_heads, forward_dim, dropout)

    def forward(self, x, value, key, src_mask, tgt_mask):
        attention_out = self.attention(x, x, x, tgt_mask)
        x = attention_out + x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.norm(x)  # output is the new query, run through the block
        x = self.block(x, key, value, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(
        self, vocab_size, emb_dim, num_layers, num_heads, forward_dim, dropout, max_len
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(emb_dim, num_heads, forward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.positional_encoding = nn.Embedding(max_len, emb_dim)
        self.output_layer = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        x = self.embedding(x)
        positions = torch.arange(x.shape[1]).expand(x.shape[0], x.shape[1]).to(device)
        x = x + self.positional_encoding(positions)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, src_mask, tgt_mask)
        x = self.output_layer(x)
        return x


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

        encoder = Encoder(
            src_vocab_size,
            emb_dim,
            num_layers,
            num_heads,
            forward_dim,
            dropout,
            max_len,
        )
        decoder = Decoder(
            tgt_vocab_size,
            emb_dim,
            num_layers,
            num_heads,
            forward_dim,
            dropout,
            max_len,
        )

        self.encoder = encoder
        self.decoder = decoder
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
        out = self.decoder(tgt, encoder_out, src_mask, tgt_mask)
        return out


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
    print("Output shape:", out.shape)  # Debug step, left in for convenience

assert (
    out.shape == expected_out_shape
), f"wrong output shape, expected: {expected_out_shape}"
