"""
Configuration for the Image Caption Transformer project.
All hyperparameters and paths are centralized here.
"""

import os
import torch

# ─── Paths ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "flickr8k")
IMAGES_DIR = os.path.join(DATA_DIR, "Images")
CAPTIONS_FILE = os.path.join(DATA_DIR, "captions.txt")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ─── Model Hyperparameters ────────────────────────────────
EMBED_DIM = 256           # Dimension of word embeddings and model hidden size
NUM_HEADS = 8             # Number of attention heads in the Transformer
NUM_DECODER_LAYERS = 3    # Number of Transformer decoder layers
DROPOUT = 0.1             # Dropout rate
ENCODER_DIM = 2048        # ResNet-50 feature dimension
NUM_PATCHES = 49          # 7x7 spatial grid from ResNet-50

# ─── Training Hyperparameters ─────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 25
GRAD_CLIP = 5.0           # Gradient clipping max norm
TEACHER_FORCING = True    # Use teacher forcing during training

# ─── Vocabulary ───────────────────────────────────────────
FREQ_THRESHOLD = 2        # Minimum word frequency to include in vocabulary
MAX_SEQ_LEN = 50          # Maximum caption length (in tokens)

# Special tokens
PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<UNK>"

# ─── Device ───────────────────────────────────────────────
# Supports: CUDA (NVIDIA), MPS (Apple Silicon M1/M2/M3), or CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Apple Silicon GPU acceleration
else:
    DEVICE = torch.device("cpu")

# ─── Image Transforms ────────────────────────────────────
IMAGE_SIZE = 224          # Input image size for ResNet

# ─── Create directories if needed ─────────────────────────
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
