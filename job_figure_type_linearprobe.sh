#!/bin/bash

#SBATCH --job-name="figure-type-linearprobe"
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

mkdir -p slurm-logs
mkdir -p /ceph/work/KLP/zihcilin39/experiments/figure_type_linearprobe

module load anaconda3/2024.10-1
ENV_NAME="ijepa"
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate $ENV_NAME

# W&B 設定
export WANDB_API_KEY="db81e71f385b3e8326d37edbb356791a25fb7389"
export WANDB_MODE="online"

echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# 安裝 wandb（確保可用）
pip install --quiet wandb

python -m src.train_figure_type \
  --config configs/figure_type_vith14_linearprobe.yaml \
  --checkpoint /ceph/work/KLP/zihcilin39/experiments/cosyn_vith14_ep300/jepa-latest.pth.tar \
  --model-name vit_huge \
  --repr last \
  --head linear \
  --log-to-wandb \
  --wandb-project ijepa-figure-type \
  --wandb-run-name figure-type-last-linear

echo "Linear probe training completed!"

