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
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.figure_type import FigureTypeDataset
from src.transforms import make_transforms
import src.models.vision_transformer as vit


def _parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str,
                               help="Path to YAML config file", default=None)

    args_config, remaining = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        parents=[config_parser], description="Figure-type classifier"
    )
    parser.add_argument("--root-dir",
                        help="Root directory containing the dataset")
    parser.add_argument("--pairs-tsv",
                        help="TSV file with <id> <figure_type> <image_path>")
    parser.add_argument("--index-json",
                        help="JSON index providing name_to_id mapping")
    parser.add_argument("--num-classes", type=int,
                        help="Number of classification labels")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint containing encoder weights")
    parser.add_argument("--model-name", default="vit_base",
                        help="Encoder model architecture")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction of data used for validation")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for train/val split")
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"],
                        default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--crop-scale", type=float, nargs=2, default=(0.3, 1.0))
    parser.add_argument("--repr", choices=["last", "last4"], default="last",
                        help="Which encoder layer representation to use")
    parser.add_argument("--head", choices=["linear", "bn_linear"],
                        default="linear", help="Classification head type")
    parser.add_argument("--output-dir", default="experiments/figure_type_linearprobe",
                        help="Directory to save checkpoints and metrics")
    parser.add_argument("--log-to-wandb", action="store_true",
                        help="Log training metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default=None,
                        help="W&B project name")
    parser.add_argument("--wandb-entity", default=None,
                        help="W&B entity/user name")
    parser.add_argument("--wandb-run-name", default=None,
                        help="W&B run name")

    if args_config.config is not None:
        import yaml

        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        dataset_cfg = cfg.get("dataset", {})
        optim_cfg = cfg.get("optimization", {})

        if isinstance(optim_cfg.get("optimizer"), list):
            optim_cfg["optimizer"] = optim_cfg["optimizer"][0]
        if isinstance(optim_cfg.get("lr"), list):
            optim_cfg["lr"] = optim_cfg["lr"][0]
        if isinstance(optim_cfg.get("weight_decay"), list):
            optim_cfg["weight_decay"] = optim_cfg["weight_decay"][0]

        parser.set_defaults(**dataset_cfg)
        parser.set_defaults(**optim_cfg)

    args = parser.parse_args(remaining)

    missing = [
        name for name in ("root_dir", "pairs_tsv", "index_json")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            f"Missing required arguments: "
            + ", ".join(f"--{m.replace('_', '-')}" for m in missing)
        )
    return args


def _build_encoder(args: argparse.Namespace) -> nn.Module:
    """Initialise encoder and load weights."""
    encoder = vit.__dict__[args.model_name]()
    state = torch.load(args.checkpoint, map_location="cpu")
    if "encoder" in state:
        state = state["encoder"]
    # Checkpoints trained with DistributedDataParallel/DataParallel prefix
    # parameter names with "module.". Strip this prefix if present so that the
    # weights can be loaded into a regular ``nn.Module`` instance.
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
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


def evaluate(
    dataloader: Iterable,
    encoder: nn.Module,
    head: nn.Module,
    device: torch.device,
    repr_mode: str,
    num_classes: int,
) -> dict:
    """Compute validation loss, accuracy and confusion matrix."""
    encoder.eval()
    head.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            feats = _extract_representation(encoder, images, repr_mode)
            logits = head(feats)
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(1)
            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)
            for t, p in zip(labels, preds):
                confusion[t, p] += 1
    per_class_acc = (
        confusion.diag().float() / confusion.sum(dim=1).clamp(min=1).float()
    ).tolist()
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "confusion_matrix": confusion.tolist(),
        "per_class_accuracy": per_class_acc,
    }


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index_path = args.index_json
    if not os.path.isabs(index_path):
        index_path = os.path.join(args.root_dir, index_path)
    if args.num_classes is not None:
        num_classes = args.num_classes
    else:
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

    # Stratified train/val split
    rng = torch.Generator().manual_seed(args.seed)
    class_to_indices: defaultdict[int, list[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in class_to_indices.values():
        indices = torch.tensor(indices)
        perm = indices[torch.randperm(len(indices), generator=rng)]
        split = int(len(perm) * (1 - args.val_ratio))
        train_indices.extend(perm[:split].tolist())
        val_indices.extend(perm[split:].tolist())

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            head.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            head.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    if args.log_to_wandb:
        import wandb

        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=args.wandb_run_name, config=vars(args))
        wandb.watch(head, log="all", log_freq=100)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            train_loader, encoder, head, optimizer, device, args.repr
        )
        val_metrics = evaluate(
            val_loader, encoder, head, device, args.repr, num_classes
        )
        print(
            f"Epoch {epoch:03d}: loss={train_loss:.4f} acc={train_acc*100:.2f}% "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']*100:.2f}%"
        )
        if args.log_to_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["accuracy"],
                },
                step=epoch,
            )
        if epoch % 10 == 0:
            save_path = os.path.join(output_dir, f"model_epoch{epoch}.pth")
            torch.save({"head": head.state_dict()}, save_path)
        scheduler.step()

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(val_metrics, f, indent=2)

if __name__ == "__main__":
    main()
