"""
Error analysis: find the best and worst predictions by sentence-level BLEU,
and compute per-length accuracy to see where the model struggles.

Usage:
    python error_analysis.py
    python error_analysis.py --checkpoint checkpoints/best_model.pth --top 10
"""

import os
import argparse
from collections import defaultdict

import pandas as pd
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from PIL import Image
from tqdm import tqdm

import config
from dataset import Vocabulary, get_transforms
from models.caption_model import CaptionModel


def run_error_analysis(model, vocab, device, top_k=5):
    df = pd.read_csv(config.CAPTIONS_FILE)
    df.columns = [c.strip() for c in df.columns]

    all_images = df["image"].unique().tolist()
    n = len(all_images)
    test_images = set(all_images[int(0.9 * n):])

    # group references
    refs_by_image = defaultdict(list)
    for _, row in df.iterrows():
        if row["image"] in test_images:
            refs_by_image[row["image"]].append(row["caption"].lower().strip().split())

    _, val_transform = get_transforms()
    smooth = SmoothingFunction().method1

    results = []

    print(f"Running error analysis on {len(refs_by_image)} test images...\n")

    for img_name in tqdm(sorted(refs_by_image.keys()), desc="Analyzing"):
        img_path = os.path.join(config.IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        image_tensor = val_transform(image).unsqueeze(0).to(device)

        gen_caption, _ = model.generate(image_tensor, vocab)
        gen_tokens = gen_caption.lower().split()
        ref_tokens = refs_by_image[img_name]

        bleu = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smooth)
        gt_text = " ".join(ref_tokens[0])

        results.append({
            "image": img_name,
            "generated": gen_caption,
            "reference": gt_text,
            "bleu": bleu,
            "gen_len": len(gen_tokens),
            "ref_len": len(ref_tokens[0]),
        })

    results.sort(key=lambda x: x["bleu"])

    # worst predictions
    print("=" * 60)
    print(f"  WORST {top_k} PREDICTIONS")
    print("=" * 60)
    for r in results[:top_k]:
        print(f"  [{r['image']}] BLEU={r['bleu']:.4f}")
        print(f"    REF: {r['reference']}")
        print(f"    GEN: {r['generated']}")
        print()

    # best predictions
    print("=" * 60)
    print(f"  BEST {top_k} PREDICTIONS")
    print("=" * 60)
    for r in results[-top_k:]:
        print(f"  [{r['image']}] BLEU={r['bleu']:.4f}")
        print(f"    REF: {r['reference']}")
        print(f"    GEN: {r['generated']}")
        print()

    # length analysis
    print("=" * 60)
    print("  BLEU BY REFERENCE CAPTION LENGTH")
    print("=" * 60)
    length_buckets = defaultdict(list)
    for r in results:
        bucket = (r["ref_len"] // 5) * 5  # group by 5
        length_buckets[bucket].append(r["bleu"])

    for bucket in sorted(length_buckets):
        scores = length_buckets[bucket]
        avg = sum(scores) / len(scores)
        print(f"  len {bucket:2d}-{bucket+4:2d}: avg BLEU={avg:.4f}  (n={len(scores)})")

    # generated length stats
    gen_lengths = [r["gen_len"] for r in results]
    ref_lengths = [r["ref_len"] for r in results]
    print(f"\n  Avg generated length: {sum(gen_lengths)/len(gen_lengths):.1f}")
    print(f"  Avg reference length: {sum(ref_lengths)/len(ref_lengths):.1f}")

    avg_bleu = sum(r["bleu"] for r in results) / len(results)
    print(f"  Mean sentence BLEU:   {avg_bleu:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    device = config.DEVICE
    vocab = torch.load(os.path.join(config.CHECKPOINT_DIR, "vocab.pth"),
                       weights_only=False)
    model, ckpt = CaptionModel.load_checkpoint(args.checkpoint, device)
    print(f"Loaded epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    run_error_analysis(model, vocab, device, top_k=args.top)


if __name__ == "__main__":
    main()
