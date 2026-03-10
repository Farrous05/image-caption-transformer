"""
Evaluation script for the Image Caption Transformer.

Computes BLEU scores on the test set and prints sample predictions.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_model.pth
"""

import argparse
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

import config
from dataset import get_data_loaders
from models.caption_model import CaptionModel


def evaluate_model(model, test_loader, vocab, device, num_samples=10):
    """
    Evaluate the model on the test set.

    Computes BLEU-1 through BLEU-4 scores and prints sample predictions.
    """
    model.eval()

    references = []   # List of lists of reference captions (tokenized)
    hypotheses = []   # List of generated captions (tokenized)
    samples = []      # Store sample predictions

    print("\nGenerating captions for test set...")

    for images, captions in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)

        for i in range(images.size(0)):
            # Generate caption
            gen_caption, _ = model.generate(images[i:i+1], vocab)

            # Ground truth caption
            gt_caption = vocab.decode(captions[i].tolist())

            # Tokenize for BLEU
            gen_tokens = gen_caption.lower().split()
            gt_tokens = gt_caption.lower().split()

            references.append([gt_tokens])  # BLEU expects list of references
            hypotheses.append(gen_tokens)

            # Collect samples
            if len(samples) < num_samples:
                samples.append((gt_caption, gen_caption))

    # ─── Compute BLEU Scores ──────────────────────────────
    smooth = SmoothingFunction().method1

    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0),
                        smoothing_function=smooth)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0),
                        smoothing_function=smooth)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0),
                        smoothing_function=smooth)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smooth)

    # ─── Print Results ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  BLEU-1: {bleu1:.4f}")
    print(f"  BLEU-2: {bleu2:.4f}")
    print(f"  BLEU-3: {bleu3:.4f}")
    print(f"  BLEU-4: {bleu4:.4f}")
    print("=" * 60)

    print(f"\n  Sample Predictions ({num_samples} examples):\n")
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

    # Load test data
    _, _, _, test_loader = get_data_loaders()

    # Evaluate
    scores = evaluate_model(model, test_loader, vocab, device,
                           num_samples=args.num_samples)

    return scores


if __name__ == "__main__":
    import os
    main()
