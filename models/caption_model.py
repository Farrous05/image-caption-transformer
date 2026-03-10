"""
Complete Image Captioning Model.

Combines the CNN Encoder (ResNet-50) and the Transformer Decoder
into a single model with both training (teacher forcing) and
inference (greedy/beam search) capabilities.
"""

import torch
import torch.nn as nn

import config
from models.encoder import Encoder
from models.decoder import TransformerDecoder


class CaptionModel(nn.Module):
    """
    Image Captioning Model = CNN Encoder + Transformer Decoder.

    Training: Uses teacher forcing — feeds ground-truth tokens as input.
    Inference: Generates tokens one at a time using greedy decoding.
    """

    def __init__(self, vocab_size, embed_dim=config.EMBED_DIM,
                 num_heads=config.NUM_HEADS, num_layers=config.NUM_DECODER_LAYERS,
                 dropout=config.DROPOUT, fine_tune_encoder=False):
        super().__init__()

        self.encoder = Encoder(embed_dim=embed_dim, fine_tune=fine_tune_encoder)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.vocab_size = vocab_size

    def forward(self, images, captions):
        """
        Forward pass for training with teacher forcing.

        Args:
            images: (batch, 3, 224, 224)
            captions: (batch, seq_len) — full caption including <START> and <END>

        Returns:
            outputs: (batch, seq_len-1, vocab_size) — predictions for each position
        """
        # Encode image → (batch, 49, embed_dim)
        encoder_out = self.encoder(images)

        # Decoder input: everything except the last token
        # Decoder target: everything except the first token (<START>)
        caption_input = captions[:, :-1]

        # Create padding mask (True where token is <PAD>=0)
        padding_mask = (caption_input == 0)

        # Decode → (batch, seq_len-1, vocab_size)
        outputs = self.decoder(caption_input, encoder_out, caption_padding_mask=padding_mask)

        return outputs

    @torch.no_grad()
    def generate(self, image, vocab, max_len=config.MAX_SEQ_LEN,
                 temperature=1.0):
        """
        Generate a caption for a single image using greedy decoding.

        Args:
            image: (1, 3, 224, 224) — single image tensor
            vocab: Vocabulary object
            max_len: maximum number of tokens to generate
            temperature: sampling temperature (1.0 = greedy)

        Returns:
            caption: string — the generated caption
            attention_weights: list of (1, 1, 49) tensors — attention per word
        """
        self.eval()
        device = next(self.parameters()).device

        # Encode image
        encoder_out = self.encoder(image)  # (1, 49, embed_dim)

        # Start with <START> token
        generated = [vocab.start_idx]
        attention_maps = []

        for _ in range(max_len):
            # Current sequence as tensor
            caption_tensor = torch.tensor([generated], dtype=torch.long, device=device)

            # Decode
            output = self.decoder(caption_tensor, encoder_out)

            # Get prediction for the last position
            logits = output[:, -1, :] / temperature  # (1, vocab_size)
            predicted_idx = logits.argmax(dim=-1).item()

            # Store attention weights for this step
            attn_weights = self.decoder.get_attention_weights()
            if attn_weights is not None:
                # Get attention for the last generated word
                attention_maps.append(attn_weights[:, -1, :].cpu())

            # Stop if we generate <END>
            if predicted_idx == vocab.end_idx:
                break

            generated.append(predicted_idx)

        # Decode token indices to words
        caption = vocab.decode(generated)

        return caption, attention_maps

    def save_checkpoint(self, filepath, epoch, optimizer, val_loss, vocab_size):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "vocab_size": vocab_size,
            "embed_dim": config.EMBED_DIM,
            "num_heads": config.NUM_HEADS,
            "num_layers": config.NUM_DECODER_LAYERS,
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load_checkpoint(cls, filepath, device=config.DEVICE):
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        model = cls(
            vocab_size=checkpoint["vocab_size"],
            embed_dim=checkpoint["embed_dim"],
            num_heads=checkpoint["num_heads"],
            num_layers=checkpoint["num_layers"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model, checkpoint
