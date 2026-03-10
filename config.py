"""All hyperparameters and paths live here."""

import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "flickr8k")
IMAGES_DIR = os.path.join(DATA_DIR, "Images")
CAPTIONS_FILE = os.path.join(DATA_DIR, "captions.txt")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# model
EMBED_DIM = 256
NUM_HEADS = 8
NUM_DECODER_LAYERS = 3
DROPOUT = 0.1
ENCODER_DIM = 2048        # resnet50 last conv block
NUM_PATCHES = 49           # 7x7

# training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 25
GRAD_CLIP = 5.0
TEACHER_FORCING = True

# vocab
FREQ_THRESHOLD = 2
MAX_SEQ_LEN = 50

PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<UNK>"

# device — prefer CUDA, then MPS (apple silicon), fallback CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

IMAGE_SIZE = 224

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
