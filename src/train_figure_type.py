# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Training script for figure-type classification with a frozen encoder.

This module performs linear classification on top of a pretrained vision
transformer encoder. Only the classification head is optimised while the
encoder remains frozen. Two options control the type of representation and
head used for classification:

* ``--repr {last, last4}`` – use the last transformer block or the average of
  the last four blocks as image representation.
* ``--head {linear, bn_linear}`` – choose between a simple linear layer or a
  BatchNorm followed by a linear layer for the classification head.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.figure_type import FigureTypeDataset
from src.transforms import make_transforms
import src.models.vision_transformer as vit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figure-type classifier")
    parser.add_argument("--root-dir", required=True,
                        help="Root directory containing the dataset")
    parser.add_argument("--pairs-tsv", required=True,
                        help="TSV file with <id> <figure_type> <image_path>")
    parser.add_argument("--index-json", required=True,
                        help="JSON index providing name_to_id mapping")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint containing encoder weights")
    parser.add_argument("--model-name", default="vit_base",
                        help="Encoder model architecture")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--crop-scale", type=float, nargs=2, default=(0.3, 1.0))
    parser.add_argument("--repr", choices=["last", "last4"], default="last",
                        help="Which encoder layer representation to use")
    parser.add_argument("--head", choices=["linear", "bn_linear"],
                        default="linear", help="Classification head type")
    return parser.parse_args()


def _build_encoder(args: argparse.Namespace) -> nn.Module:
    """Initialise encoder and load weights."""
    encoder = vit.__dict__[args.model_name]()
    state = torch.load(args.checkpoint, map_location="cpu")
    if "encoder" in state:
        state = state["encoder"]
    encoder.load_state_dict(state, strict=True)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    return encoder


def _build_head(embed_dim: int, num_classes: int, head_type: str) -> nn.Module:
    if head_type == "linear":
        head = nn.Linear(embed_dim, num_classes)
    else:
        head = nn.Sequential(nn.BatchNorm1d(embed_dim),
                             nn.Linear(embed_dim, num_classes))
    return head
def _extract_representation(encoder: nn.Module, x: torch.Tensor,
                            repr_mode: str) -> torch.Tensor:
    """Encode images and obtain a single vector representation."""
    h = encoder.patch_embed(x)
    h = h + encoder.interpolate_pos_encoding(h, encoder.pos_embed)
    outputs: list[torch.Tensor] = []
    for blk in encoder.blocks:
        h = blk(h)
        out = encoder.norm(h) if encoder.norm is not None else h
        outputs.append(out)
    if repr_mode == "last":
        h = outputs[-1]
    else:
        h = torch.stack(outputs[-4:], dim=0).mean(0)
    h = h.mean(dim=1)
    return h


def train_one_epoch(
    dataloader: Iterable,
    encoder: nn.Module,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    repr_mode: str,
) -> tuple[float, float]:
    encoder.eval()
    head.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            feats = _extract_representation(encoder, images, repr_mode)
        logits = head(feats)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += images.size(0)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index_path = args.index_json
    if not os.path.isabs(index_path):
        index_path = os.path.join(args.root_dir, index_path)
    with open(index_path, "r") as f:
        index_data = json.load(f)
    num_classes = len(index_data.get("name_to_id", {}))

    encoder = _build_encoder(args).to(device)
    head = _build_head(encoder.embed_dim, num_classes, args.head).to(device)

    transform = make_transforms(
        crop_size=args.crop_size,
        crop_scale=tuple(args.crop_scale),
    )
    dataset = FigureTypeDataset(
        root_dir=args.root_dir,
        pairs_tsv=args.pairs_tsv,
        index_json=args.index_json,
        split="train",
        transform=transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(
            dataloader, encoder, head, optimizer, device, args.repr
        )
        print(f"Epoch {epoch:03d}: loss={loss:.4f} acc={acc*100:.2f}%")

if __name__ == "__main__":
    main()
