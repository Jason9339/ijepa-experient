#!/bin/bash

echo "Adding W&B integration to I-JEPA..."

# 1. 更新配置文件，添加 W&B 設定
cat > configs/cosyn_vith14_ep300.yaml << 'YAML_EOF'
data:
  batch_size: 32
  color_jitter_strength: 0.0
  crop_scale:
  - 0.3
  - 1.0
  crop_size: 224
  image_folder: ""
  num_workers: 4
  pin_mem: true
  root_path: /ceph/work/KLP/zihcilin39/datasets
  use_color_distortion: false
  use_gaussian_blur: false
  use_horizontal_flip: false
logging:
  folder: /ceph/work/KLP/zihcilin39/experiments/cosyn_vith14_ep300/
  write_tag: jepa
  # W&B 設定
  use_wandb: true
  wandb_project: "ijepa-cosyn-diagrams"
  wandb_entity: null
  wandb_run_name: "cosyn-vith14-ep300"
  wandb_tags: ["ijepa", "cosyn", "diagram", "vit-huge"]
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
YAML_EOF

echo "✓ Updated configs/cosyn_vith14_ep300.yaml with W&B settings"

# 2. 創建 W&B 整合的作業腳本
cat > job_cosyn_wandb.sh << 'SCRIPT_EOF'
#!/bin/bash

#SBATCH --job-name="ijepa-cosyn-wandb"
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

# 設定 W&B API Key
export WANDB_API_KEY="db81e71f385b3e8326d37edbb356791a25fb7389"
export WANDB_MODE="online"

# 確認環境
echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# 安裝 wandb
pip install wandb

# 修改 main.py 來使用 train_wandb.py
sed -i 's/from src.train import main as app_main/from src.train_wandb import main as app_main/' main.py

# 執行 I-JEPA 訓練
python main.py --fname configs/cosyn_vith14_ep300.yaml --devices cuda:0

echo "Training completed!"
SCRIPT_EOF

chmod +x job_cosyn_wandb.sh
echo "✓ Created job_cosyn_wandb.sh with W&B integration"

echo "W&B integration setup complete!"
