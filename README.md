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
| **Training** | Teacher forcing, cross-entropy loss, Adam optimizer, LR scheduling |
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
python train.py --resume checkpoints/best_model.pth --epochs 25  # Resume training
```

### 4. Evaluate

```bash
python evaluate.py
```

Computes BLEU-1 through BLEU-4 scores on the test set.

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

Upload an image to get a caption and attention heatmaps showing where the model looked for each word.

## Project Structure

```
image-caption-transformer/
├── config.py              # Hyperparameters and paths
├── dataset.py             # Vocabulary, Dataset, DataLoaders
├── train.py               # Training loop
├── evaluate.py            # BLEU score evaluation
├── visualize.py           # Attention heatmap visualization
├── app.py                 # Flask web demo
├── models/
│   ├── encoder.py         # ResNet-50 feature extractor
│   ├── decoder.py         # Transformer decoder with cross-attention
│   └── caption_model.py   # Combined encoder-decoder model
├── templates/
│   └── index.html         # Web demo UI
├── static/
│   └── style.css          # Web demo styling
├── data/
│   └── download.py        # Dataset download instructions
└── requirements.txt
```

## Key Concepts

- **Transfer Learning** — Pre-trained CNN as feature extractor
- **Transformer Architecture** — Self-attention, cross-attention, positional encoding
- **Multimodal Learning** — Bridging computer vision and natural language processing
- **Model Interpretability** — Attention visualization
- **ML Engineering** — Training pipeline with checkpointing, evaluation, and serving

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

## License

MIT
