"""
Attention visualization for the Image Caption Transformer.

Generates attention heatmaps showing which parts of the image
the model focuses on when generating each word of the caption.

Usage:
    python visualize.py
    python visualize.py --image path/to/image.jpg
    python visualize.py --checkpoint checkpoints/best_model.pth
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

import config
from models.caption_model import CaptionModel


def get_inference_transform():
    """Get image transform for inference."""
    return T.Compose([
        T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_and_preprocess_image(image_path):
    """Load an image and return both the original and preprocessed versions."""
    original_image = Image.open(image_path).convert("RGB")

    transform = get_inference_transform()
    image_tensor = transform(original_image).unsqueeze(0)  # (1, 3, 224, 224)

    return original_image, image_tensor


def visualize_attention(image_path, model, vocab, device, save_path=None):
    """
    Generate a caption and visualize attention heatmaps.

    Creates a grid showing the original image and attention maps
    for each word in the generated caption.
    """
    # Load image
    original_image, image_tensor = load_and_preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    # Generate caption and get attention weights
    caption, attention_maps = model.generate(image_tensor, vocab)
    words = caption.split()

    if not attention_maps:
        print("Warning: No attention weights captured.")
        return caption

    print(f"  Image: {os.path.basename(image_path)}")
    print(f"  Caption: {caption}")

    # Create visualization grid
    num_words = len(words)
    cols = min(num_words + 1, 6)  # Max 6 columns
    rows = (num_words + 1 + cols - 1) // cols  # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]

    # Flatten axes for easy indexing
    axes_flat = [ax for row in axes for ax in row]

    # Hide all axes first
    for ax in axes_flat:
        ax.axis("off")

    # Show original image
    axes_flat[0].imshow(original_image)
    axes_flat[0].set_title("Original", fontsize=10, fontweight="bold")

    # Show attention heatmap for each word
    original_np = np.array(original_image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))

    for i, (word, attn) in enumerate(zip(words, attention_maps)):
        if i + 1 >= len(axes_flat):
            break

        ax = axes_flat[i + 1]

        # Reshape attention: (1, 49) → (7, 7)
        attn_map = attn.squeeze().numpy()  # (49,)
        attn_map = attn_map.reshape(7, 7)

        # Upscale to image size
        attn_resized = np.array(
            Image.fromarray(attn_map).resize(
                (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BICUBIC
            )
        )

        # Normalize to [0, 1]
        attn_resized = (attn_resized - attn_resized.min()) / (
            attn_resized.max() - attn_resized.min() + 1e-8
        )

        # Show original image with attention overlay
        ax.imshow(original_np)
        ax.imshow(attn_resized, alpha=0.6, cmap="jet")
        ax.set_title(f'"{word}"', fontsize=10, fontweight="bold", color="#333")

    plt.suptitle(f'Caption: "{caption}"', fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved to: {save_path}")
    else:
        plt.show()

    plt.close()
    return caption


def visualize_samples(model, vocab, device, num_samples=5, save_dir=config.OUTPUT_DIR):
    """Visualize attention for random images from the test set."""
    os.makedirs(save_dir, exist_ok=True)

    # Get sample images from the dataset
    images_dir = config.IMAGES_DIR
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

    if not image_files:
        print("Error: No images found in data directory.")
        return

    # Select random samples
    import random
    samples = random.sample(image_files, min(num_samples, len(image_files)))

    print(f"\nGenerating attention visualizations for {len(samples)} images...\n")

    for i, img_name in enumerate(samples, 1):
        img_path = os.path.join(images_dir, img_name)
        save_path = os.path.join(save_dir, f"attention_{i}_{img_name.replace('.jpg', '.png')}")
        visualize_attention(img_path, model, vocab, device, save_path=save_path)
        print()


def main():
    parser = argparse.ArgumentParser(description="Visualize Attention Heatmaps")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a specific image to caption")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
                        help="Path to model checkpoint")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of random samples to visualize")
    parser.add_argument("--output-dir", type=str, default=config.OUTPUT_DIR,
                        help="Directory to save visualizations")
    args = parser.parse_args()

    device = config.DEVICE
    print(f"Device: {device}")

    # Load vocabulary
    vocab_path = os.path.join(config.CHECKPOINT_DIR, "vocab.pth")
    vocab = torch.load(vocab_path, weights_only=False)

    # Load model
    print(f"  Loading checkpoint: {args.checkpoint}")
    model, checkpoint = CaptionModel.load_checkpoint(args.checkpoint, device)
    print(f"  Loaded model from epoch {checkpoint['epoch']}")

    if args.image:
        # Visualize a specific image
        save_path = os.path.join(args.output_dir, "attention_custom.png")
        visualize_attention(args.image, model, vocab, device, save_path=save_path)
    else:
        # Visualize random samples
        visualize_samples(model, vocab, device, num_samples=args.num_samples,
                         save_dir=args.output_dir)


if __name__ == "__main__":
    main()
