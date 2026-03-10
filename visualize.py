"""
Attention heatmap visualization.

Overlays cross-attention weights on the input image to show
which regions the model looks at for each generated word.
"""

import os
import argparse
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

import config
from models.caption_model import CaptionModel


def get_inference_transform():
    return T.Compose([
        T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_and_preprocess_image(image_path):
    original_image = Image.open(image_path).convert("RGB")
    transform = get_inference_transform()
    image_tensor = transform(original_image).unsqueeze(0)
    return original_image, image_tensor


def visualize_attention(image_path, model, vocab, device, save_path=None):
    """Generate a caption and plot attention heatmaps per word."""
    original_image, image_tensor = load_and_preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    caption, attention_maps = model.generate(image_tensor, vocab)
    words = caption.split()

    if not attention_maps:
        print("Warning: no attention weights captured")
        return caption

    print(f"  Image: {os.path.basename(image_path)}")
    print(f"  Caption: {caption}")

    num_words = len(words)
    cols = min(num_words + 1, 6)
    rows = (num_words + 1 + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]

    axes_flat = [ax for row in axes for ax in row]

    for ax in axes_flat:
        ax.axis("off")

    axes_flat[0].imshow(original_image)
    axes_flat[0].set_title("Original", fontsize=10, fontweight="bold")

    original_np = np.array(original_image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))

    for i, (word, attn) in enumerate(zip(words, attention_maps)):
        if i + 1 >= len(axes_flat):
            break

        ax = axes_flat[i + 1]

        attn_map = attn.squeeze().numpy().reshape(7, 7)

        attn_resized = np.array(
            Image.fromarray(attn_map).resize(
                (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BICUBIC
            )
        )
        # normalize to [0, 1]
        attn_resized = (attn_resized - attn_resized.min()) / (
            attn_resized.max() - attn_resized.min() + 1e-8
        )

        ax.imshow(original_np)
        ax.imshow(attn_resized, alpha=0.6, cmap="jet")
        ax.set_title(f'"{word}"', fontsize=10, fontweight="bold", color="#333")

    plt.suptitle(f'Caption: "{caption}"', fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved to: {save_path}")
    else:
        plt.show()

    plt.close()
    return caption


def visualize_samples(model, vocab, device, num_samples=5, save_dir=config.OUTPUT_DIR):
    os.makedirs(save_dir, exist_ok=True)

    image_files = [f for f in os.listdir(config.IMAGES_DIR) if f.endswith(".jpg")]
    if not image_files:
        print("No images found.")
        return

    samples = random.sample(image_files, min(num_samples, len(image_files)))

    print(f"\nGenerating attention visualizations for {len(samples)} images...\n")

    for i, img_name in enumerate(samples, 1):
        img_path = os.path.join(config.IMAGES_DIR, img_name)
        save_path = os.path.join(save_dir, f"attention_{i}_{img_name.replace('.jpg', '.png')}")
        visualize_attention(img_path, model, vocab, device, save_path=save_path)
        print()


def main():
    parser = argparse.ArgumentParser(description="Visualize attention heatmaps")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=config.OUTPUT_DIR)
    args = parser.parse_args()

    device = config.DEVICE
    print(f"Device: {device}")

    vocab_path = os.path.join(config.CHECKPOINT_DIR, "vocab.pth")
    vocab = torch.load(vocab_path, weights_only=False)

    print(f"  Loading checkpoint: {args.checkpoint}")
    model, checkpoint = CaptionModel.load_checkpoint(args.checkpoint, device)
    print(f"  Loaded model from epoch {checkpoint['epoch']}")

    if args.image:
        save_path = os.path.join(args.output_dir, "attention_custom.png")
        visualize_attention(args.image, model, vocab, device, save_path=save_path)
    else:
        visualize_samples(model, vocab, device, num_samples=args.num_samples,
                         save_dir=args.output_dir)


if __name__ == "__main__":
    main()
