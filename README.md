# Image Caption Transformer

Generate natural language descriptions of images using a **CNN Encoder (ResNet-50)** and a **Transformer Decoder** with cross-attention, plus attention heatmap visualization showing where the model looks when generating each word.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Architecture

```
Image (224x224) -> ResNet-50 (pretrained) -> 49 spatial features (7x7 grid)
                                                    |
                                                    v
<START> token -> Word Embedding + Positional Encoding -> Transformer Decoder -> Caption
                                                    ^
                                          Cross-Attention (words <-> image regions)
```

| Component | Details |
|-----------|---------|
| **Encoder** | ResNet-50 (pretrained on ImageNet), outputs 49x2048 feature grid, projected to 256-dim |
| **Decoder** | 3-layer Transformer with 8-head attention, masked self-attention + cross-attention |
| **Training** | Teacher forcing, cross-entropy loss, Adam optimizer (weight_decay=1e-5), StepLR scheduling, early stopping |
| **Inference** | Greedy decoding with attention weight extraction |

## Quick Start

### 1. Setup

```bash
git clone https://github.com/Farrous05/image-caption-transformer.git
cd image-caption-transformer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Dataset

Download [Flickr8k from Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) and extract to `data/flickr8k/`:

```
data/flickr8k/
├── Images/          # 8091 JPEG images
└── captions.txt     # Image-caption pairs
```

### 3. Train

```bash
python train.py                    # Full training (25 epochs)
python train.py --epochs 15        # Shorter run
python train.py --debug            # Smoke test (5 batches/epoch)
python train.py --resume checkpoints/best_model.pth --epochs 25  # Resume
```

### 4. Evaluate

```bash
python evaluate.py                 # BLEU-1 to BLEU-4 (multi-reference)
python error_analysis.py           # Best/worst predictions + per-length breakdown
```

### 5. Visualize Attention

```bash
python visualize.py                         # Random samples
python visualize.py --image path/to/img.jpg # Specific image
```

### 6. Web Demo

```bash
python app.py
# Open http://localhost:5000
```

### 7. Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Project Structure

```
image-caption-transformer/
├── config.py              # Hyperparameters and paths
├── dataset.py             # Vocabulary, Dataset, DataLoaders
├── train.py               # Training loop with early stopping
├── evaluate.py            # BLEU score evaluation (multi-reference)
├── error_analysis.py      # Per-image error analysis
├── visualize.py           # Attention heatmap visualization
├── app.py                 # Flask web demo
├── models/
│   ├── encoder.py         # ResNet-50 feature extractor
│   ├── decoder.py         # Transformer decoder with cross-attention
│   └── caption_model.py   # Combined encoder-decoder model
├── tests/
│   └── test_model.py      # Unit tests for vocab, encoder, decoder, model
├── templates/
│   └── index.html         # Web demo UI
├── static/
│   └── style.css          # Web demo styling
├── data/
│   └── download.py        # Dataset download helper
└── requirements.txt       # Pinned dependencies
```

## Configuration

Edit `config.py` to adjust hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBED_DIM` | 256 | Model hidden dimension |
| `NUM_HEADS` | 8 | Attention heads |
| `NUM_DECODER_LAYERS` | 3 | Transformer layers |
| `BATCH_SIZE` | 32 | Training batch size |
| `LEARNING_RATE` | 1e-4 | Initial learning rate |
| `EPOCHS` | 25 | Training epochs |
| `DROPOUT` | 0.1 | Dropout rate |
| `GRAD_CLIP` | 5.0 | Gradient clipping max norm |

## Results

### BLEU Scores (Test Set — 810 images, multi-reference)

| Metric | Score |
|--------|-------|
| BLEU-1 | 0.6278 |
| BLEU-2 | 0.4339 |
| BLEU-3 | 0.3064 |
| BLEU-4 | 0.2103 |

Trained for 25 epochs on an Apple M1 (MPS backend). Best checkpoint at epoch 18 (val_loss = 2.7434).

### Evaluation Bug Fix

The initial evaluation script compared each generated caption against **a single reference caption**, which is not the standard protocol. Flickr8k provides 5 human-written captions per image. Switching to multi-reference BLEU (the standard) revealed that the model was performing much better than originally measured:

| Metric | Single-ref (broken) | Multi-ref (fixed) | Change |
|--------|--------------------:|------------------:|-------:|
| BLEU-1 | 0.3742 | 0.6278 | +68% |
| BLEU-2 | 0.2014 | 0.4339 | +115% |
| BLEU-3 | 0.1230 | 0.3064 | +149% |
| BLEU-4 | 0.0747 | 0.2103 | +181% |

The fix groups all 5 reference captions per image in `evaluate.py` using `build_test_references()`, instead of using whichever single caption happened to be in the batch. This is a common pitfall in captioning evaluation — BLEU is designed to compare against multiple valid references, and using only one heavily penalizes correct but differently-worded captions.

### Other Improvements Made During Debugging

While investigating the low scores, several other issues were found and fixed:

- **No reproducibility seeds** — added `set_seed()` pinning `torch`, `numpy`, `random`, and `cudnn`
- **`--debug` flag was declared but never used** — now limits training to 5 batches/epoch for smoke testing
- **No weight decay** — added `weight_decay=1e-5` to Adam optimizer for regularization
- **No early stopping** — added `--patience` flag (default 5 epochs) to stop training when validation loss plateaus
- **Missing `pandas` in requirements.txt** — added it and pinned all dependency versions
- **No tests** — added 9 unit tests covering vocabulary, encoder, decoder, and full model
- **No error analysis** — added `error_analysis.py` showing best/worst predictions and BLEU-by-caption-length breakdown

## Reproducibility

Training seeds are set via `--seed` (default 42). This pins `torch`, `numpy`, `random`, and `cudnn` for reproducible results on the same hardware. Note that results may still vary slightly across different GPU architectures.

Data split is deterministic: 80% train / 10% val / 10% test by unique image name ordering in the CSV.

## Limitations & Bias

- **English only** — Flickr8k captions are all in English, so the model cannot generate captions in other languages.
- **Western-centric imagery** — Flickr8k is biased towards photos uploaded by English-speaking Flickr users, so the model may perform poorly on images from underrepresented cultures or regions.
- **Simple tokenization** — whitespace splitting with no subword tokenization means the model cannot handle rare or compound words well.
- **Greedy decoding** — beam search would likely improve caption quality but is not implemented.
- **Small dataset** — Flickr8k has only ~8k images. Larger datasets like COCO Captions (330k images) would yield better results.

## Licenses

| Component | License |
|-----------|---------|
| This repository | MIT |
| Flickr8k dataset | [Creative Commons](https://www.kaggle.com/datasets/adityajn105/flickr8k) — images subject to original Flickr photographer licenses |
| ResNet-50 (torchvision) | BSD 3-Clause |
