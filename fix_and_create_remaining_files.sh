#!/bin/bash

echo "Moving cosyn.py to correct location and creating remaining files..."

# 1. 移動 cosyn.py 到正確位置
if [ -f "src/cosyn.py" ]; then
    mv src/cosyn.py src/datasets/cosyn.py
    echo "✓ Moved cosyn.py to src/datasets/"
fi

# 重新創建正確的 cosyn.py
cat > src/datasets/cosyn.py << 'EOF'
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
EOF

echo "✓ Created/Updated src/datasets/cosyn.py"

# 2. 創建 CoSyn 配置文件
cat > configs/cosyn_vith14_ep300.yaml << 'EOF'
data:
  batch_size: 32  # 從小一點開始測試
  color_jitter_strength: 0.0
  crop_scale:
  - 0.3
  - 1.0
  crop_size: 224
  image_folder: ""  # CoSyn diagram 不需要這個參數
  num_workers: 4   # 減少工作進程數
  pin_mem: true
  root_path: /ceph/work/KLP/zihcilin39/datasets  # 你的資料集路徑
  use_color_distortion: false
  use_gaussian_blur: false
  use_horizontal_flip: false
logging:
  folder: /ceph/work/KLP/zihcilin39/experiments/cosyn_vith14_ep300/
  write_tag: jepa
mask:
  allow_overlap: false
  aspect_ratio:
  - 0.75
  - 1.5
  enc_mask_scale:
  - 0.85
  - 1.0
  min_keep: 10
  num_enc_masks: 1
  num_pred_masks: 4
  patch_size: 14
  pred_mask_scale:
  - 0.15
  - 0.2
meta:
  copy_data: false
  load_checkpoint: false
  model_name: vit_huge
  pred_depth: 12
  pred_emb_dim: 384
  read_checkpoint: null
  use_bfloat16: true
optimization:
  ema:
  - 0.996
  - 1.0
  epochs: 300
  final_lr: 1.0e-06
  final_weight_decay: 0.4
  ipe_scale: 1.0
  lr: 0.001
  start_lr: 0.0002
  warmup: 40
  weight_decay: 0.04
EOF

echo "✓ Created configs/cosyn_vith14_ep300.yaml"

# 3. 創建訓練作業腳本
cat > job_cosyn.sh << 'EOF'
#!/bin/bash

#SBATCH --job-name="ijepa-cosyn-training"
#SBATCH --partition=a100-al9
#SBATCH --reservation=klp_reservation
#SBATCH --qos=klp_a100_reservation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH -o slurm-logs/%j.out
#SBATCH -e slurm-logs/%j.err

set -e

# 建立 log 檔的資料夾
mkdir -p slurm-logs
mkdir -p /ceph/work/KLP/zihcilin39/experiments/cosyn_vith14_ep300

# 載入 anaconda
module load anaconda3/2024.10-1

# 設定要使用的 conda 環境名稱
ENV_NAME="ijepa"

# 初始化 conda
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

# 啟用 conda 環境
conda activate $ENV_NAME

# 確認環境
echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# 執行 I-JEPA 訓練
python main.py --fname configs/cosyn_vith14_ep300.yaml --devices cuda:0

echo "Training completed!"
EOF

chmod +x job_cosyn.sh
echo "✓ Created job_cosyn.sh"

echo ""
echo "Files created successfully!"
echo ""
echo "Checking file structure:"
echo "src/datasets/ contents:"
ls -la src/datasets/
echo ""
echo "configs/ contents:"
ls -la configs/ | grep cosyn
