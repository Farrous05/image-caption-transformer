"""Dataset and vocabulary for Flickr8k."""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from collections import Counter
import torchvision.transforms as T

import config


class Vocabulary:
    """Maps words <-> indices. Built from caption strings."""

    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold

        self.pad_idx = 0
        self.start_idx = 1
        self.end_idx = 2
        self.unk_idx = 3

        self.word2idx = {
            config.PAD_TOKEN: self.pad_idx,
            config.START_TOKEN: self.start_idx,
            config.END_TOKEN: self.end_idx,
            config.UNK_TOKEN: self.unk_idx,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)

    def build_vocabulary(self, captions):
        counter = Counter()
        for caption in captions:
            tokens = self.tokenize(caption)
            counter.update(tokens)

        for word, count in counter.items():
            if count >= self.freq_threshold:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        print(f"  Vocabulary built: {len(self)} words (threshold={self.freq_threshold})")

    @staticmethod
    def tokenize(text):
        return text.lower().strip().split()

    def numericalize(self, text):
        tokens = self.tokenize(text)
        return (
            [self.start_idx]
            + [self.word2idx.get(t, self.unk_idx) for t in tokens]
            + [self.end_idx]
        )

    def decode(self, indices):
        words = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            word = self.idx2word.get(idx, config.UNK_TOKEN)
            if word == config.END_TOKEN:
                break
            if word not in (config.START_TOKEN, config.PAD_TOKEN):
                words.append(word)
        return " ".join(words)


class FlickrDataset(Dataset):
    """Loads image-caption pairs for a given split (train/val/test)."""

    def __init__(self, captions_file, images_dir, vocab, transform=None, split="train"):
        self.images_dir = images_dir
        self.vocab = vocab
        self.transform = transform

        df = pd.read_csv(captions_file)
        df.columns = [c.strip() for c in df.columns]

        all_images = df["image"].unique().tolist()
        n = len(all_images)

        # 80/10/10 split by image
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

        if split == "train":
            split_images = set(all_images[:train_end])
        elif split == "val":
            split_images = set(all_images[train_end:val_end])
        elif split == "test":
            split_images = set(all_images[val_end:])
        else:
            raise ValueError(f"Unknown split: {split}")

        mask = df["image"].isin(split_images)
        self.df = df[mask].reset_index(drop=True)

        print(f"  {split.upper()} split: {len(split_images)} images, {len(self.df)} captions")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["image"])
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        caption = torch.tensor(self.vocab.numericalize(row["caption"]), dtype=torch.long)
        return image, caption


class CaptionCollate:
    """Pads captions to the same length within a batch."""

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images, captions = zip(*batch)
        images = torch.stack(images, dim=0)
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        return images, captions


def get_transforms():
    train_transform = T.Compose([
        T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def get_data_loaders():
    """Build vocab and return (vocab, train_loader, val_loader, test_loader)."""
    print("Loading Flickr8k dataset...")

    df = pd.read_csv(config.CAPTIONS_FILE)
    df.columns = [c.strip() for c in df.columns]
    all_captions = df["caption"].tolist()

    vocab = Vocabulary(freq_threshold=config.FREQ_THRESHOLD)
    vocab.build_vocabulary(all_captions)

    train_transform, val_transform = get_transforms()

    train_dataset = FlickrDataset(
        config.CAPTIONS_FILE, config.IMAGES_DIR, vocab,
        transform=train_transform, split="train"
    )
    val_dataset = FlickrDataset(
        config.CAPTIONS_FILE, config.IMAGES_DIR, vocab,
        transform=val_transform, split="val"
    )
    test_dataset = FlickrDataset(
        config.CAPTIONS_FILE, config.IMAGES_DIR, vocab,
        transform=val_transform, split="test"
    )

    collate = CaptionCollate(pad_idx=vocab.pad_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True,
        collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True,
        collate_fn=collate
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True,
        collate_fn=collate
    )

    print(f"  DataLoaders ready (batch_size={config.BATCH_SIZE})")
    return vocab, train_loader, val_loader, test_loader
