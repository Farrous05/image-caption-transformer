"""
Transformer Decoder for image captioning.

Generates captions word-by-word using:
- Masked self-attention (so each word only sees previous words)
- Cross-attention (so each word can look at the image features)
- Feed-forward network

Stores cross-attention weights for visualization.
"""

import math
import torch
import torch.nn as nn

import config


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Adds position information to word embeddings so the Transformer
    knows the order of words in the sequence.
    """

    def __init__(self, embed_dim, max_len=config.MAX_SEQ_LEN, dropout=config.DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)

        # Register as buffer (not a parameter, but moves with model to GPU)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        Returns:
            x + positional encoding, with dropout
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DecoderLayer(nn.Module):
    """
    Single Transformer decoder layer.

    Flow: Masked Self-Attention → Cross-Attention → Feed-Forward

    The cross-attention weights are stored for visualization.
    """

    def __init__(self, embed_dim, num_heads, ff_dim=None, dropout=config.DROPOUT):
        super().__init__()

        if ff_dim is None:
            ff_dim = embed_dim * 4

        # Masked self-attention (caption attends to itself)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(embed_dim)

        # Cross-attention (caption attends to image features)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(embed_dim)

        # Store attention weights for visualization
        self.cross_attn_weights = None

    def forward(self, x, encoder_out, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Args:
            x: (batch, seq_len, embed_dim) — caption embeddings
            encoder_out: (batch, 49, embed_dim) — image features
            tgt_mask: (seq_len, seq_len) — causal mask
            tgt_key_padding_mask: (batch, seq_len) — padding mask

        Returns:
            x: (batch, seq_len, embed_dim)
        """
        # 1. Masked self-attention (+ residual connection + layer norm)
        attn_out, _ = self.self_attn(
            x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        x = self.self_attn_norm(x + attn_out)

        # 2. Cross-attention over image features (+ residual + norm)
        cross_out, cross_weights = self.cross_attn(
            query=x, key=encoder_out, value=encoder_out
        )
        self.cross_attn_weights = cross_weights  # Save for visualization
        x = self.cross_attn_norm(x + cross_out)

        # 3. Feed-forward (+ residual + norm)
        ff_out = self.ff(x)
        x = self.ff_norm(x + ff_out)

        return x


class TransformerDecoder(nn.Module):
    """
    Full Transformer decoder for caption generation.

    Architecture:
        Word Embedding + Positional Encoding
        → N × DecoderLayer (self-attn → cross-attn → FF)
        → Linear projection to vocabulary size
    """

    def __init__(self, vocab_size, embed_dim=config.EMBED_DIM,
                 num_heads=config.NUM_HEADS, num_layers=config.NUM_DECODER_LAYERS,
                 dropout=config.DROPOUT):
        super().__init__()

        self.embed_dim = embed_dim

        # Word embedding + positional encoding
        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)

        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Final projection to vocabulary
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _generate_causal_mask(self, seq_len, device):
        """
        Generate a causal (look-ahead) mask.
        Prevents the decoder from attending to future tokens.

        Returns:
            mask: (seq_len, seq_len) with -inf for future positions
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, captions, encoder_out, caption_padding_mask=None):
        """
        Args:
            captions: (batch, seq_len) — token indices
            encoder_out: (batch, 49, embed_dim) — image features
            caption_padding_mask: (batch, seq_len) — True where padded

        Returns:
            output: (batch, seq_len, vocab_size) — logits for each position
        """
        seq_len = captions.size(1)

        # Embed words and add positional encoding
        x = self.word_embedding(captions) * math.sqrt(self.embed_dim)
        x = self.pos_encoding(x)

        # Causal mask
        causal_mask = self._generate_causal_mask(seq_len, captions.device)

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_out, tgt_mask=causal_mask,
                      tgt_key_padding_mask=caption_padding_mask)

        # Project to vocabulary
        output = self.fc_out(x)

        return output

    def get_attention_weights(self):
        """
        Get the cross-attention weights from the last decoder layer.

        Returns:
            weights: (batch, seq_len, 49) — attention over image patches
        """
        return self.layers[-1].cross_attn_weights
