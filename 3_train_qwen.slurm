#!/bin/bash
#SBATCH --job-name=qwen_chess_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-a100-lowbig
#SBATCH --output=logs_train_%j.log

module purge
module load gcc/14.2.0
module load python/3.13.0-gcc14.2.0
module load cuda/12.8.0-gcc14.2.0

# 激活安装有 unsloth, trl, torch 的环境
source ~/scratch/llmchess/bin/activate

# 为防止 HuggingFace 缓存冲突导致的问题
export HF_HOME="/users/sgzwa126/.cache/huggingface"

echo "Starting Unsloth Training Pipeline..."
torchrun --nproc_per_node=2 train_qwen.py