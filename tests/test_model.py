"""Basic sanity checks for the model components."""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.encoder import Encoder
from models.decoder import TransformerDecoder
from models.caption_model import CaptionModel
from dataset import Vocabulary


VOCAB_SIZE = 100
BATCH = 2
SEQ_LEN = 10


class TestVocabulary:
    def test_build_and_lookup(self):
        vocab = Vocabulary(freq_threshold=1)
        vocab.build_vocabulary(["a dog runs", "a cat sits", "a dog sits"])
        assert len(vocab) > 4  # at least special tokens + some words
        assert vocab.word2idx["dog"] == vocab.word2idx["dog"]

    def test_numericalize_roundtrip(self):
        vocab = Vocabulary(freq_threshold=1)
        vocab.build_vocabulary(["hello world", "hello there"])
        indices = vocab.numericalize("hello world")
        assert indices[0] == vocab.start_idx
        assert indices[-1] == vocab.end_idx
        decoded = vocab.decode(indices)
        assert decoded == "hello world"

    def test_unk_token(self):
        vocab = Vocabulary(freq_threshold=5)
        vocab.build_vocabulary(["cat"])
        indices = vocab.numericalize("cat")
        # "cat" appears only once, threshold is 5 -> should be UNK
        assert vocab.unk_idx in indices


class TestEncoder:
    def test_output_shape(self):
        encoder = Encoder(embed_dim=64)
        images = torch.randn(BATCH, 3, 224, 224)
        out = encoder(images)
        assert out.shape == (BATCH, 49, 64)

    def test_frozen_by_default(self):
        encoder = Encoder(embed_dim=64, fine_tune=False)
        for p in encoder.resnet.parameters():
            assert not p.requires_grad


class TestDecoder:
    def test_output_shape(self):
        decoder = TransformerDecoder(vocab_size=VOCAB_SIZE, embed_dim=64,
                                     num_heads=4, num_layers=1)
        captions = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        encoder_out = torch.randn(BATCH, 49, 64)
        out = decoder(captions, encoder_out)
        assert out.shape == (BATCH, SEQ_LEN, VOCAB_SIZE)

    def test_attention_weights_stored(self):
        decoder = TransformerDecoder(vocab_size=VOCAB_SIZE, embed_dim=64,
                                     num_heads=4, num_layers=2)
        captions = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        encoder_out = torch.randn(BATCH, 49, 64)
        decoder(captions, encoder_out)
        weights = decoder.get_attention_weights()
        assert weights is not None
        assert weights.shape == (BATCH, SEQ_LEN, 49)


class TestCaptionModel:
    def test_forward_shape(self):
        model = CaptionModel(vocab_size=VOCAB_SIZE, embed_dim=64,
                              num_heads=4, num_layers=1)
        images = torch.randn(BATCH, 3, 224, 224)
        captions = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        out = model(images, captions)
        # output seq is one shorter (teacher forcing shifts by 1)
        assert out.shape == (BATCH, SEQ_LEN - 1, VOCAB_SIZE)

    def test_generate_returns_string(self):
        vocab = Vocabulary(freq_threshold=1)
        vocab.build_vocabulary(["a dog runs fast", "a cat sits down"])
        model = CaptionModel(vocab_size=len(vocab), embed_dim=64,
                              num_heads=4, num_layers=1)
        image = torch.randn(1, 3, 224, 224)
        caption, attn = model.generate(image, vocab, max_len=10)
        assert isinstance(caption, str)
        assert isinstance(attn, list)
