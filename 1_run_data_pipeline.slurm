#!/bin/bash
#SBATCH --job-name=chess_data_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=160
#SBATCH --time=12:00:00
#SBATCH --partition=lowpriority
#SBATCH --output=logs_pipeline_%j.log

module purge
module load gcc/14.2.0
module load python/3.13.0-gcc14.2.0

source ~/scratch/llmchess/bin/activate

export GCC_LIB_DIR="$(dirname $(dirname $(which gcc)))/lib64"
export LD_LIBRARY_PATH="$GCC_LIB_DIR:$LD_LIBRARY_PATH"

export STOCKFISH_PATH="/users/sgzwa126/stockfish/stockfish-ubuntu-x86-64-avx2"
export DATA_DIR="/users/sgzwa126/llmchess/data"
export DEFAULT_DEPTH=15
mkdir -p $DATA_DIR

# ---------------- 分片架构配置 ----------------
SHARDS=8                     # 启动 8 套独立流水线
ENGINES_PER_SHARD=18         # 每套配备 18 个引擎 (总共 144 个引擎)
GAMES_PER_SHARD=625         # 每套生成 625 局 (总计 5000 局)
CONCURRENCY_PER_SHARD=25     # 每套 25 个并发对局压榨引擎

echo "=================================================="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | CPUs: 160"
echo "🚀 启动分布式多进程架构: $SHARDS 个独立分片"
echo "=================================================="

# 1. 批量启动 8 个独立的 Server
for i in $(seq 1 $SHARDS); do
    export SOCKET_PATH="/tmp/stockfish_${SLURM_JOB_ID}_${i}.sock"
    export NUM_ENGINES=$ENGINES_PER_SHARD

    # 后台静默启动，日志重定向防止控制台刷屏
    python stockfish_server.py > ${DATA_DIR}/server_${i}.log 2>&1 &
    SERVER_PIDS[$i]=$!
done

echo "[1/3] 正在唤醒全部 144 个引擎进程，请等待 15 秒..."
sleep 15
echo "✅ 所有 Server 已就绪！"

# 2. 批量启动 8 个独立的数据生成客户端
echo "[2/3] 全面开火！八大分片开始极速生成自我对弈数据..."
for i in $(seq 1 $SHARDS); do
    export SOCKET_PATH="/tmp/stockfish_${SLURM_JOB_ID}_${i}.sock"
    export NUM_GAMES=$GAMES_PER_SHARD
    export CONCURRENCY=$CONCURRENCY_PER_SHARD
    # 指定各自独立的数据输出文件，防止并发写入冲突
    export SFT_DATA_PATH="${DATA_DIR}/sft_data_${i}.jsonl"
    export RL_REWARD_PATH="${DATA_DIR}/rl_rewards_${i}.json"

    python generate_data.py > ${DATA_DIR}/client_${i}.log 2>&1 &
    CLIENT_PIDS[$i]=$!
done

# 3. 阻塞主脚本，等待 8 个客户端全部生成完毕
for pid in ${CLIENT_PIDS[*]}; do
    wait $pid
done

echo "✅ 50000 局数据生成完毕，正在关闭 Server 集群..."
for pid in ${SERVER_PIDS[*]}; do
    kill -SIGTERM $pid
done

# 4. 自动合并数据
echo "[3/3] 正在合并 8 个分片的数据..."
# 合并 SFT 数据 (直接追加即可)
cat ${DATA_DIR}/sft_data_*.jsonl > ${DATA_DIR}/sft_data.jsonl

# 使用内联 Python 脚本合并 8 个 JSON 字典
python3 -c "
import json, glob
merged_dict = {}
print('开始合并 RL 字典...')
for f in glob.glob('${DATA_DIR}/rl_rewards_*.json'):
    with open(f, 'r', encoding='utf-8') as file:
        merged_dict.update(json.load(file))
with open('${DATA_DIR}/rl_rewards.json', 'w', encoding='utf-8') as out:
    json.dump(merged_dict, out, ensure_ascii=False, separators=(',', ':'))
print(f'成功合并 {len(merged_dict)} 个独特局面的强化学习价值。')
"

# 清理碎片文件
rm -f ${DATA_DIR}/sft_data_*.jsonl
rm -f ${DATA_DIR}/rl_rewards_*.json
rm -f ${DATA_DIR}/server_*.log ${DATA_DIR}/client_*.log

echo "🎉 完美收官！最终训练数据已保存在 $DATA_DIR"