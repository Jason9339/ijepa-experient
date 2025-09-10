# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import csv
import json
from typing import Optional, Callable

from logging import getLogger
from PIL import Image

import torch
from torch.utils.data import Dataset

logger = getLogger()


class FigureTypeDataset(Dataset):
    """Dataset for figure-type classification.

    Parameters
    ----------
    root_dir : str
        Root directory containing images.
    pairs_tsv : str
        Path to TSV file listing ``id``, ``figure_type`` and ``image_path``.
        Can be relative to ``root_dir``.
    index_json : str
        JSON file containing ``name_to_id`` mapping and optional ``splits``.
        Can be relative to ``root_dir``.
    split : str
        Which data split to load.  If ``index_json`` provides a ``splits``
        dictionary the dataset will be filtered accordingly.  Otherwise the
        parameter is ignored and all samples are loaded.
    transform : Optional[Callable]
        Optional transform to apply on a PIL image.
    """

    def __init__(
        self,
        root_dir: str,
        pairs_tsv: str,
        index_json: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # type: list[tuple[str, int]]

        # Resolve paths relative to root_dir if necessary
        if not os.path.isabs(pairs_tsv):
            pairs_tsv = os.path.join(root_dir, pairs_tsv)
        if not os.path.isabs(index_json):
            index_json = os.path.join(root_dir, index_json)

        # Load index information
        with open(index_json, "r") as f:
            index_data = json.load(f)
        name_to_id = index_data.get("name_to_id", {})

        # Determine ids for requested split if available
        split_ids = None
        splits = index_data.get("splits")
        if isinstance(splits, dict) and split in splits:
            split_ids = set(splits[split])
        elif split in index_data:
            # Some index files might have split keys at the top level
            try:
                split_ids = set(index_data[split])
            except Exception:
                split_ids = None

        # Parse pairs file and build samples list
        with open(pairs_tsv, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                sample_id = row["id"]
                if split_ids is not None and sample_id not in split_ids:
                    continue
                label_name = row["figure_type"]
                label_id = name_to_id.get(label_name)
                if label_id is None:
                    # Skip unknown labels
                    continue
                img_path = row.get("image_path", "")
                if img_path == "":
                    # fallback to images/<id>.png if path missing
                    img_path = os.path.join("images", f"{sample_id}.png")
                full_path = (
                    img_path
                    if os.path.isabs(img_path)
                    else os.path.join(root_dir, img_path)
                )
                self.samples.append((full_path, int(label_id)))

        logger.info(
            "Loaded %d samples from %s (split=%s)",
            len(self.samples),
            pairs_tsv,
            split,
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, index: int):  # type: ignore[override]
        path, label = self.samples[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        else:
            # Convert to tensor without external deps
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
            img = img.view(img.height, img.width, 3).permute(2, 0, 1)
            img = img.float().div(255.0)
        return img, label
