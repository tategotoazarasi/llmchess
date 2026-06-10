#!/bin/bash
#SBATCH --job-name=chess_engine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --output=logs_serve_%j.log

module purge
module load python/3.11.9-gcc14.2.0
module load cuda/12.8.0-gcc14.2.0

source ~/scratch/sgl_env/bin/activate

export NCCL_NET_GDR_LEVEL=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

MODEL_DIR="/users/sgzwa126/llmchess/chess_model_rl_final"

sglang serve \
  --model-path "$MODEL_DIR" \
  --served-model-name chess-engine \
  --chat-template chatml \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static 0.5 \
  --trust-remote-code