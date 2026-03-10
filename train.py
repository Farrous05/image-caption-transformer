"""
Training script for the Image Caption Transformer.

Usage:
    python train.py                 # Full training
    python train.py --epochs 1 --debug  # Quick smoke test
"""

import os
import argparse
import time
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from dataset import get_data_loaders
from models.caption_model import CaptionModel


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]", leave=False)
    for images, captions in pbar:
        images = images.to(device)
        captions = captions.to(device)

        # Forward pass
        # Output: (batch, seq_len-1, vocab_size)
        # Target: captions[:, 1:] (everything after <START>)
        outputs = model(images, captions)
        targets = captions[:, 1:]

        # Reshape for cross-entropy: (batch*seq_len, vocab_size) vs (batch*seq_len)
        outputs = outputs.reshape(-1, outputs.size(-1))
        targets = targets.reshape(-1)

        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(model, val_loader, criterion, device, vocab, epoch):
    """Validate the model and print sample captions."""
    model.eval()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [VAL]  ", leave=False)
    sample_images = None

    for images, captions in pbar:
        images = images.to(device)
        captions = captions.to(device)

        outputs = model(images, captions)
        targets = captions[:, 1:]

        outputs_flat = outputs.reshape(-1, outputs.size(-1))
        targets_flat = targets.reshape(-1)

        loss = criterion(outputs_flat, targets_flat)

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Save first batch for sample captions
        if sample_images is None:
            sample_images = images[:3]
            sample_captions = captions[:3]

    avg_loss = total_loss / num_batches

    # Print sample captions
    if sample_images is not None:
        print(f"\n  Sample captions (Epoch {epoch}):")
        for i in range(min(3, sample_images.size(0))):
            # Ground truth
            gt_caption = vocab.decode(sample_captions[i].tolist())
            # Generated
            gen_caption, _ = model.generate(sample_images[i:i+1], vocab)
            print(f"     GT:  {gt_caption}")
            print(f"     GEN: {gen_caption}")
            print()

    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train Image Caption Transformer")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help=f"Number of epochs (default: {config.EPOCHS})")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                        help=f"Learning rate (default: {config.LEARNING_RATE})")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help=f"Batch size (default: {config.BATCH_SIZE})")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: use small subset of data")
    args = parser.parse_args()

    # Override config with CLI args
    if args.batch_size != config.BATCH_SIZE:
        config.BATCH_SIZE = args.batch_size

    device = config.DEVICE
    print(f"Device: {device}")
    print(f"Config: embed_dim={config.EMBED_DIM}, heads={config.NUM_HEADS}, "
          f"layers={config.NUM_DECODER_LAYERS}")
    print(f"Training: epochs={args.epochs}, lr={args.lr}, "
          f"batch_size={args.batch_size}")
    print()

    # ─── Data ─────────────────────────────────────────────
    vocab, train_loader, val_loader, _ = get_data_loaders()
    vocab_size = len(vocab)
    print(f"  Vocab size: {vocab_size}")
    print()

    # ─── Model ────────────────────────────────────────────
    model = CaptionModel(
        vocab_size=vocab_size,
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_DECODER_LAYERS,
        dropout=config.DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print()

    # ─── Loss & Optimizer ─────────────────────────────────
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    start_epoch = 1
    best_val_loss = float("inf")

    # Resume from checkpoint
    if args.resume:
        print(f"  Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["val_loss"]
        print(f"  Resumed from epoch {start_epoch - 1}")

    # Save vocab for later use
    vocab_path = os.path.join(config.CHECKPOINT_DIR, "vocab.pth")
    torch.save(vocab, vocab_path)
    print(f"  Vocabulary saved to: {vocab_path}")

    # ─── Training Loop ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING START")
    print("=" * 60 + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device, vocab, epoch)

        elapsed = time.time() - start_time
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{args.epochs} │ "
              f"Train Loss: {train_loss:.4f} │ "
              f"Val Loss: {val_loss:.4f} │ "
              f"LR: {current_lr:.6f} │ "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            model.save_checkpoint(best_path, epoch, optimizer, val_loss, vocab_size)
            print(f"  * New best model saved (val_loss={val_loss:.4f})")

        # Save periodic checkpoint
        if epoch % 5 == 0:
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch{epoch}.pth")
            model.save_checkpoint(ckpt_path, epoch, optimizer, val_loss, vocab_size)

    print("\n" + "=" * 60)
    print(f"  TRAINING COMPLETE — Best Val Loss: {best_val_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
