# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import datasets
from logging import getLogger
import torch
from PIL import Image

logger = getLogger()


def make_cosyn(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,  # 添加這個參數以兼容原始接口，但不使用
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None
):
    dataset = CoSynDataset(
        root=root_path,
        transform=transform,
        train=training)
    
    logger.info('CoSyn dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('CoSyn data loader created')

    return dataset, data_loader, dist_sampler


class CoSynDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        transform=None,
        train=True
    ):
        """
        CoSyn Dataset wrapper for diagram data only
        
        :param root: root directory containing CoSyn/diagram data
        :param transform: image transforms to apply
        :param train: whether to load train data (for now, we use all data)
        """
        self.root = root
        self.transform = transform
        
        # 直接載入 diagram 資料夾
        diagram_path = os.path.join(root, "CoSyn", "diagram")
        logger.info(f'Loading CoSyn diagram dataset from {diagram_path}')
        
        try:
            self.dataset = datasets.load_from_disk(diagram_path)
            logger.info(f'Loaded {len(self.dataset)} diagram samples from CoSyn dataset')
            
            # 檢查資料集結構
            sample = self.dataset[0]
            logger.info(f'Dataset features: {list(sample.keys())}')
            if 'image' in sample:
                logger.info(f'Image type: {type(sample["image"])}')
            
        except Exception as e:
            logger.error(f'Failed to load CoSyn diagram dataset: {e}')
            raise

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        
        # 取得圖像
        if isinstance(sample['image'], Image.Image):
            img = sample['image']
        else:
            # 如果是其他格式，嘗試轉換
            img = sample['image']
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
        
        # 確保是 RGB 格式
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 應用轉換
        if self.transform is not None:
            img = self.transform(img)
        
        # 為了與 ImageNet 載入器兼容，返回圖像和一個虛擬標籤
        return img, 0  # 0 是虛擬標籤，因為 I-JEPA 是無監督學習
