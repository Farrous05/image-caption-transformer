"""Encoder + Decoder combined into one model."""

import torch
import torch.nn as nn

import config
from models.encoder import Encoder
from models.decoder import TransformerDecoder


class CaptionModel(nn.Module):
    """
    CNN Encoder (ResNet-50) + Transformer Decoder.
    Uses teacher forcing for training, greedy decoding at inference.
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
        encoder_out = self.encoder(images)

        # input is everything except last token, target is everything except <START>
        caption_input = captions[:, :-1]
        padding_mask = (caption_input == 0)

        outputs = self.decoder(caption_input, encoder_out, caption_padding_mask=padding_mask)
        return outputs

    @torch.no_grad()
    def generate(self, image, vocab, max_len=config.MAX_SEQ_LEN, temperature=1.0):
        """Greedy decoding for a single image. Returns (caption_str, attention_maps)."""
        self.eval()
        device = next(self.parameters()).device

        encoder_out = self.encoder(image)

        generated = [vocab.start_idx]
        attention_maps = []

        for _ in range(max_len):
            caption_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            output = self.decoder(caption_tensor, encoder_out)

            logits = output[:, -1, :] / temperature
            predicted_idx = logits.argmax(dim=-1).item()

            attn_weights = self.decoder.get_attention_weights()
            if attn_weights is not None:
                attention_maps.append(attn_weights[:, -1, :].cpu())

            if predicted_idx == vocab.end_idx:
                break

            generated.append(predicted_idx)

        caption = vocab.decode(generated)
        return caption, attention_maps

    def save_checkpoint(self, filepath, epoch, optimizer, val_loss, vocab_size):
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
