"""
Transformer Decoder for captioning.

self-attn (masked) -> cross-attn (over image features) -> FFN
Stores cross-attention weights so we can visualize them later.
"""

import math
import torch
import torch.nn as nn

import config


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al.)"""

    def __init__(self, embed_dim, max_len=config.MAX_SEQ_LEN, dropout=config.DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DecoderLayer(nn.Module):
    """Single transformer decoder layer."""

    def __init__(self, embed_dim, num_heads, ff_dim=None, dropout=config.DROPOUT):
        super().__init__()

        if ff_dim is None:
            ff_dim = embed_dim * 4

        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(embed_dim)

        self.cross_attn_weights = None  # saved for visualization

    def forward(self, x, encoder_out, tgt_mask=None, tgt_key_padding_mask=None):
        # masked self-attention + residual
        attn_out, _ = self.self_attn(
            x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        x = self.self_attn_norm(x + attn_out)

        # cross-attention over image patches + residual
        cross_out, cross_weights = self.cross_attn(
            query=x, key=encoder_out, value=encoder_out
        )
        self.cross_attn_weights = cross_weights
        x = self.cross_attn_norm(x + cross_out)

        # feed-forward + residual
        ff_out = self.ff(x)
        x = self.ff_norm(x + ff_out)

        return x


class TransformerDecoder(nn.Module):
    """
    Word Embedding + Positional Encoding
    -> N x DecoderLayer
    -> Linear to vocab_size
    """

    def __init__(self, vocab_size, embed_dim=config.EMBED_DIM,
                 num_heads=config.NUM_HEADS, num_layers=config.NUM_DECODER_LAYERS,
                 dropout=config.DROPOUT):
        super().__init__()

        self.embed_dim = embed_dim
        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _generate_causal_mask(self, seq_len, device):
        """Upper-triangular mask filled with -inf to block future tokens."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, captions, encoder_out, caption_padding_mask=None):
        seq_len = captions.size(1)

        # scale embeddings by sqrt(d) like in the original paper
        x = self.word_embedding(captions) * math.sqrt(self.embed_dim)
        x = self.pos_encoding(x)

        causal_mask = self._generate_causal_mask(seq_len, captions.device)

        for layer in self.layers:
            x = layer(x, encoder_out, tgt_mask=causal_mask,
                      tgt_key_padding_mask=caption_padding_mask)

        output = self.fc_out(x)
        return output

    def get_attention_weights(self):
        """Return cross-attention weights from the last layer (for visualization)."""
        return self.layers[-1].cross_attn_weights
