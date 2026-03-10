"""
Evaluation script for the Image Caption Transformer.

Computes BLEU scores on the test set using all 5 reference captions
per image (standard protocol for Flickr8k).

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_model.pth
"""

import os
import argparse
from collections import defaultdict

import pandas as pd
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from PIL import Image
from tqdm import tqdm

import config
from dataset import Vocabulary, get_transforms
from models.caption_model import CaptionModel


def build_test_references(captions_file):
    """
    Group all reference captions by image for proper BLEU evaluation.

    Flickr8k has 5 captions per image -- using all of them gives
    a fairer score than comparing against a single random caption.
    """
    df = pd.read_csv(captions_file)
    df.columns = [c.strip() for c in df.columns]

    all_images = df["image"].unique().tolist()
    n = len(all_images)
    test_images = set(all_images[int(0.9 * n):])  # last 10%

    refs_by_image = defaultdict(list)
    for _, row in df.iterrows():
        if row["image"] in test_images:
            tokens = row["caption"].lower().strip().split()
            refs_by_image[row["image"]].append(tokens)

    return refs_by_image


def evaluate_model(model, refs_by_image, vocab, device, num_samples=10):
    """
    Evaluate on the test set with multi-reference BLEU.
    """
    model.eval()

    _, val_transform = get_transforms()
    references = []
    hypotheses = []
    samples = []

    print(f"\nGenerating captions for {len(refs_by_image)} test images...")

    image_names = sorted(refs_by_image.keys())

    for img_name in tqdm(image_names, desc="Evaluating"):
        img_path = os.path.join(config.IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        image_tensor = val_transform(image).unsqueeze(0).to(device)

        gen_caption, _ = model.generate(image_tensor, vocab)
        gen_tokens = gen_caption.lower().split()

        ref_tokens = refs_by_image[img_name]  # list of 5 tokenized refs

        references.append(ref_tokens)
        hypotheses.append(gen_tokens)

        if len(samples) < num_samples:
            gt_text = " ".join(ref_tokens[0])
            samples.append((gt_text, gen_caption))

    # Compute BLEU
    smooth = SmoothingFunction().method1

    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0),
                        smoothing_function=smooth)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0),
                        smoothing_function=smooth)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0),
                        smoothing_function=smooth)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smooth)

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  BLEU-1: {bleu1:.4f}")
    print(f"  BLEU-2: {bleu2:.4f}")
    print(f"  BLEU-3: {bleu3:.4f}")
    print(f"  BLEU-4: {bleu4:.4f}")
    print(f"  (using {len(references)} images, multi-reference)")
    print("=" * 60)

    print(f"\n  Sample Predictions ({len(samples)} examples):\n")
    for i, (gt, gen) in enumerate(samples, 1):
        print(f"  [{i}]")
        print(f"    GT:  {gt}")
        print(f"    GEN: {gen}")
        print()

    return {"bleu1": bleu1, "bleu2": bleu2, "bleu3": bleu3, "bleu4": bleu4}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Image Caption Transformer")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
                        help="Path to model checkpoint")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of sample predictions to show")
    args = parser.parse_args()

    device = config.DEVICE
    print(f"Device: {device}")

    # Load vocabulary
    vocab_path = os.path.join(config.CHECKPOINT_DIR, "vocab.pth")
    vocab = torch.load(vocab_path, weights_only=False)
    print(f"  Vocab size: {len(vocab)}")

    # Load model
    print(f"  Loading checkpoint: {args.checkpoint}")
    model, checkpoint = CaptionModel.load_checkpoint(args.checkpoint, device)
    print(f"  Loaded model from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.4f})")

    # Build multi-reference test set
    refs_by_image = build_test_references(config.CAPTIONS_FILE)
    print(f"  Test images: {len(refs_by_image)}")

    # Evaluate
    scores = evaluate_model(model, refs_by_image, vocab, device,
                           num_samples=args.num_samples)

    return scores


if __name__ == "__main__":
    main()
