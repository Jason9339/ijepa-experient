#!/bin/bash

#SBATCH --job-name="test-cosyn-dataset"
#SBATCH --partition=a100-al9
#SBATCH --reservation=klp_reservation
#SBATCH --qos=klp_a100_reservation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00  # 30分鐘測試
#SBATCH --mem=16G
#SBATCH -o slurm-logs/%j.out
#SBATCH -e slurm-logs/%j.err

set -e

# 建立 log 檔的資料夾
mkdir -p slurm-logs

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

# 安裝需要的套件
pip install datasets

# 測試資料集載入
echo "Testing CoSyn diagram dataset..."
python test_cosyn_dataset.py

echo "Dataset test completed!"
