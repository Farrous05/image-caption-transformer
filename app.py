"""
Flask web demo for the Image Caption Transformer.

Upload an image → get a caption + attention heatmap.

Usage:
    python app.py
    Then open http://localhost:5000
"""

import os
import io
import base64
import json
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

import config
from models.caption_model import CaptionModel
from visualize import get_inference_transform

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

# Global model and vocab (loaded once)
model = None
vocab = None
device = config.DEVICE


def load_model():
    """Load the trained model and vocabulary."""
    global model, vocab

    vocab_path = os.path.join(config.CHECKPOINT_DIR, "vocab.pth")
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No model checkpoint found at {checkpoint_path}. "
            "Please train the model first with: python train.py"
        )

    vocab = torch.load(vocab_path, weights_only=False)
    model, _ = CaptionModel.load_checkpoint(checkpoint_path, device)
    model.eval()
    print(f"Model loaded on {device}")


def generate_attention_image(original_image, attention_maps, words):
    """Generate a single image with attention heatmaps for each word."""
    num_words = min(len(words), len(attention_maps))
    if num_words == 0:
        return None

    original_np = np.array(original_image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))

    cols = min(num_words, 5)
    rows = (num_words + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    axes_flat = [ax for row in axes for ax in row]
    for ax in axes_flat:
        ax.axis("off")

    for i in range(num_words):
        ax = axes_flat[i]
        attn_map = attention_maps[i].squeeze().numpy().reshape(7, 7)
        attn_resized = np.array(
            Image.fromarray(attn_map).resize(
                (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BICUBIC
            )
        )
        attn_resized = (attn_resized - attn_resized.min()) / (
            attn_resized.max() - attn_resized.min() + 1e-8
        )
        ax.imshow(original_np)
        ax.imshow(attn_resized, alpha=0.6, cmap="jet")
        ax.set_title(f'"{words[i]}"', fontsize=9, fontweight="bold")

    plt.tight_layout()

    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#1a1a2e")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    return img_base64


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/caption", methods=["POST"])
def caption():
    """Generate a caption for an uploaded image."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Load and preprocess image
        original_image = Image.open(file.stream).convert("RGB")
        transform = get_inference_transform()
        image_tensor = transform(original_image).unsqueeze(0).to(device)

        # Generate caption
        caption_text, attention_maps = model.generate(image_tensor, vocab)
        words = caption_text.split()

        # Generate attention visualization
        attn_image = generate_attention_image(original_image, attention_maps, words)

        # Convert original image to base64
        buf = io.BytesIO()
        original_image.save(buf, format="PNG")
        buf.seek(0)
        original_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return jsonify({
            "caption": caption_text,
            "original_image": original_base64,
            "attention_image": attn_image,
            "words": words,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Loading Image Caption Transformer...")
    load_model()
    print("Starting web server at http://localhost:5000")
    app.run(debug=False, port=5000)
