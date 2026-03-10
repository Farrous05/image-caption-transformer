"""
Training script for the Image Caption Transformer.

Usage:
    python train.py
    python train.py --epochs 1 --debug
"""

import os
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from dataset import get_data_loaders
from models.caption_model import CaptionModel


def set_seed(seed=42):
    """Pin all random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch,
                    max_batches=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]", leave=False)
    for images, captions in pbar:
        images = images.to(device)
        captions = captions.to(device)

        # Output: (batch, seq_len-1, vocab_size)
        # Target: captions[:, 1:] — everything after <START>
        outputs = model(images, captions)
        targets = captions[:, 1:]

        # Reshape for cross-entropy: (batch*seq_len, vocab_size) vs (batch*seq_len)
        outputs = outputs.reshape(-1, outputs.size(-1))
        targets = targets.reshape(-1)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if max_batches and num_batches >= max_batches:
            break

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(model, val_loader, criterion, device, vocab, epoch,
             max_batches=None):
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

        if sample_images is None:
            sample_images = images[:3]
            sample_captions = captions[:3]

        if max_batches and num_batches >= max_batches:
            break

    avg_loss = total_loss / num_batches

    if sample_images is not None:
        print(f"\n  Sample captions (Epoch {epoch}):")
        for i in range(min(3, sample_images.size(0))):
            gt_caption = vocab.decode(sample_captions[i].tolist())
            gen_caption, _ = model.generate(sample_images[i:i+1], vocab)
            print(f"     GT:  {gt_caption}")
            print(f"     GEN: {gen_caption}")
            print()

    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train Image Caption Transformer")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                        help="Quick smoke test: only 5 batches per epoch")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.batch_size != config.BATCH_SIZE:
        config.BATCH_SIZE = args.batch_size

    device = config.DEVICE
    print(f"Device: {device}")
    print(f"Config: embed_dim={config.EMBED_DIM}, heads={config.NUM_HEADS}, "
          f"layers={config.NUM_DECODER_LAYERS}, seed={args.seed}")
    print(f"Training: epochs={args.epochs}, lr={args.lr}, "
          f"batch_size={args.batch_size}, patience={args.patience}")
    if args.debug:
        print("  ** DEBUG MODE: 5 batches per epoch **")
    print()

    # Data
    vocab, train_loader, val_loader, _ = get_data_loaders()
    vocab_size = len(vocab)
    print(f"  Vocab size: {vocab_size}\n")

    # Model
    model = CaptionModel(
        vocab_size=vocab_size,
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_DECODER_LAYERS,
        dropout=config.DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable\n")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    start_epoch = 1
    best_val_loss = float("inf")
    epochs_no_improve = 0

    if args.resume:
        print(f"  Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["val_loss"]
        print(f"  Resumed from epoch {start_epoch - 1}")

    vocab_path = os.path.join(config.CHECKPOINT_DIR, "vocab.pth")
    torch.save(vocab, vocab_path)

    max_batches = 5 if args.debug else None

    print("\n" + "=" * 60)
    print("  TRAINING START")
    print("=" * 60 + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            max_batches=max_batches,
        )
        val_loss = validate(
            model, val_loader, criterion, device, vocab, epoch,
            max_batches=max_batches,
        )

        elapsed = time.time() - start_time
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"{elapsed:.1f}s")

        # Save best model + early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            model.save_checkpoint(best_path, epoch, optimizer, val_loss, vocab_size)
            print(f"  * Best model saved (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"  Early stopping: no improvement for {args.patience} epochs")
                break

        if epoch % 5 == 0:
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch{epoch}.pth")
            model.save_checkpoint(ckpt_path, epoch, optimizer, val_loss, vocab_size)

    print("\n" + "=" * 60)
    print(f"  TRAINING COMPLETE — Best Val Loss: {best_val_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
