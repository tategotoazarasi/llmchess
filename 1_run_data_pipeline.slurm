#!/bin/bash
#SBATCH --job-name=chess_data_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=160
#SBATCH --time=12:00:00
#SBATCH --partition=nodes
#SBATCH --output=logs_pipeline_%j.log

module purge
module load gcc/14.2.0
module load python/3.13.0-gcc14.2.0

source ~/scratch/llmchess/bin/activate

export GCC_LIB_DIR="$(dirname $(dirname $(which gcc)))/lib64"
export LD_LIBRARY_PATH="$GCC_LIB_DIR:$LD_LIBRARY_PATH"

# ---------------- 配置区 ----------------
# 【这里已经改回你原本指定的二进制文件路径】
export STOCKFISH_PATH="/users/sgzwa126/stockfish/stockfish-ubuntu-x86-64-avx2"

# 使用 SLURM_JOB_ID 动态生成 socket 路径，防止同一个节点运行多个任务时冲突
export SOCKET_PATH="/tmp/stockfish_server_${SLURM_JOB_ID}.sock"

export DEFAULT_DEPTH=15
export NUM_ENGINES=150
export NUM_GAMES=5000
export CONCURRENCY=200

echo "=================================================="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "=================================================="

# 1. 在后台启动 Stockfish Server
echo "[1/3] 正在后台启动 Stockfish Server..."
python stockfish_server.py &
SERVER_PID=$!

# 2. 轮询等待 Server 创建 Socket 文件并就绪
echo "[2/3] 等待 Server 初始化..."
while [ ! -S "$SOCKET_PATH" ]; do
    # 如果进程挂了，直接退出避免死循环
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ Server 启动失败，请查看日志。"
        exit 1
    fi
    sleep 1
done

# 额外等待 3 秒确保所有引擎完全就绪
sleep 3
echo "✅ Server 已就绪！"

# 3. 在前台运行数据生成客户端
echo "[3/3] 开始生成自我对弈数据..."
python generate_data.py

# 4. 数据生成结束后，优雅地清理后台的 Server 进程
echo "数据生成完毕，正在关闭 Server..."
kill -SIGTERM $SERVER_PID
wait $SERVER_PID
echo "🎉 整个流水线执行完毕！"