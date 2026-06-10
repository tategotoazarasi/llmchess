#!/usr/bin/env python3
"""
Chess LLM Data Generator (Async UNIX Socket Client - 极速并发修复版)
===================================================
通过连接 Stockfish UNIX Socket Server 进行自我对弈生成训练数据。

修复说明：
- 移除了单点串行锁，现在每个对局拥有独立的 Socket 连接，完美打满 15 个引擎的 CPU！
- 增加了文件 I/O 的实时的 flush()，支持 tail -f 即时观察数据。
- 增加了 RL 字典的定时自动保存机制，防止跑了十几个小时被杀导致数据丢失。
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import sys
from typing import Any, Dict, List

try:
    import chess
except ImportError:
    sys.exit("致命错误：缺少依赖，请执行 'pip install chess'")

# ──────────────────────────────────────────────────────────────────────────────
# 配置参数
# ──────────────────────────────────────────────────────────────────────────────

SOCKET_PATH = os.getenv("SOCKET_PATH", "/tmp/stockfish_server.sock")
DATA_DIR = "/users/sgzwa126/llmchess/data"
NUM_GAMES = int(os.getenv("NUM_GAMES", "5000"))  # 要生成的对局数量
CONCURRENCY = int(os.getenv("CONCURRENCY", "100"))  # 异步并发对局数
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.15"))  # Softmax 采样温度

os.makedirs(DATA_DIR, exist_ok=True)
SFT_DATA_PATH = os.getenv("SFT_DATA_PATH", os.path.join(DATA_DIR, "sft_data.jsonl"))
RL_REWARD_PATH = os.getenv("RL_REWARD_PATH", os.path.join(DATA_DIR, "rl_rewards.json"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(module)s] %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 核心数据生成逻辑
# ──────────────────────────────────────────────────────────────────────────────


def softmax_sample(moves_data: List[Dict[str, Any]], temperature: float) -> str:
    """根据胜率使用 Softmax 进行 Off-policy 采样，返回选定的 UCI 着法"""
    if temperature <= 0.0:
        return moves_data[0]["move"]

    logits = [m["win_prob"] / temperature for m in moves_data]
    max_logit = max(logits)
    exp_logits = [math.exp(l - max_logit) for l in logits]
    sum_exp = sum(exp_logits)
    probs = [e / sum_exp for e in exp_logits]

    rand_val = random.random()
    cumulative = 0.0
    for move_info, prob in zip(moves_data, probs):
        cumulative += prob
        if rand_val <= cumulative:
            return move_info["move"]
    return moves_data[-1]["move"]


def build_chatml_prompt(fen: str) -> str:
    """构建 ChatML 格式的用户提示"""
    return (
        f"You are a Grandmaster chess engine. Given the current board state in FEN:\n"
        f"{fen}\n"
        f"Analyze the board and output the absolute best move in standard UCI format. "
        f"Output ONLY the UCI string (e.g., e2e4)."
    )


async def play_game(
    socket_path: str, sft_file, rl_dict: dict, rl_lock: asyncio.Lock
) -> None:
    """运行单局自我对弈：每个对局独享一个极速内存 Socket 连接"""
    try:
        reader, writer = await asyncio.open_unix_connection(socket_path)
    except Exception as e:
        log.error("无法连接到 Server: %s", e)
        return

    board = chess.Board()
    max_plies = 150  # 防止死循环

    try:
        for _ in range(max_plies):
            if board.is_game_over():
                break

            fen = board.fen()

            # 1. 组装并发送请求
            req = {"fen": fen}
            raw_req = json.dumps(req, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
            writer.write(len(raw_req).to_bytes(4, "big") + raw_req)
            await writer.drain()

            # 2. 接收响应
            try:
                hdr = await reader.readexactly(4)
                size = int.from_bytes(hdr, "big")
                raw_resp = await reader.readexactly(size)
            except asyncio.IncompleteReadError:
                break  # 连接异常断开

            resp = json.loads(raw_resp.decode("utf-8"))

            if "error" in resp or resp.get("game_over", False):
                break

            moves = resp.get("moves", [])
            if not moves:
                break

            # 3. 记录 SFT 数据 (仅最优解)
            best_move = moves[0]["move"]
            sft_record = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a highly capable AI chess player.",
                    },
                    {"role": "user", "content": build_chatml_prompt(fen)},
                    {"role": "assistant", "content": best_move},
                ]
            }
            # 写入并立即 flush！让你用 tail -f 立刻能看到数据
            sft_file.write(json.dumps(sft_record, ensure_ascii=False) + "\n")
            sft_file.flush()

            # 4. 记录 RL 字典 (所有动作及其 Advantage)
            async with rl_lock:
                if fen not in rl_dict:
                    rl_dict[fen] = {}
                    for m in moves:
                        rl_dict[fen][m["move"]] = m["win_prob_delta"]

            # 5. Off-policy 采样下一步
            chosen_move_uci = softmax_sample(moves, TEMPERATURE)
            board.push(chess.Move.from_uci(chosen_move_uci))

    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# 主控协程
# ──────────────────────────────────────────────────────────────────────────────


async def amain() -> None:
    log.info("开始并发生成数据...")
    rl_dict = {}
    rl_lock = asyncio.Lock()

    sft_file = open(SFT_DATA_PATH, "w", encoding="utf-8")

    tasks = set()
    completed = 0

    for i in range(NUM_GAMES):
        if len(tasks) >= CONCURRENCY:
            # 等待任一对局结束
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            completed += len(done)
            log.info("✅ 进度: %d / %d 局对弈已完成...", completed, NUM_GAMES)

            # 每完成 20 局增量保存一次 RL 字典（防止意外中断白跑）
            if completed % 20 == 0:
                async with rl_lock:
                    with open(RL_REWARD_PATH, "w", encoding="utf-8") as f:
                        json.dump(rl_dict, f, ensure_ascii=False, separators=(",", ":"))

        # 抛出新对局任务
        task = asyncio.create_task(play_game(SOCKET_PATH, sft_file, rl_dict, rl_lock))
        tasks.add(task)

    # 等待剩余任务完成
    if tasks:
        done, _ = await asyncio.wait(tasks)
        completed += len(done)
        log.info("✅ 进度: %d / %d 局对弈已完成...", completed, NUM_GAMES)

    sft_file.close()

    # 最终完整写入 RL 奖励字典
    log.info("正在保存最终 RL 奖励字典 (包含 %d 个唯一局面状态)...", len(rl_dict))
    with open(RL_REWARD_PATH, "w", encoding="utf-8") as f:
        json.dump(rl_dict, f, ensure_ascii=False, separators=(",", ":"))

    log.info("🎉 数据生成完毕！")
    log.info("   SFT 数据路径: %s", SFT_DATA_PATH)
    log.info("   RL 字典路径: %s", RL_REWARD_PATH)


if __name__ == "__main__":
    asyncio.run(amain())
