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
