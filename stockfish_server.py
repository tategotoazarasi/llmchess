#!/usr/bin/env python3
"""
Stockfish UNIX Socket Server — RL Training Edition (MultiPV 极速优化版)
======================================================================
为强化学习训练提供 Stockfish 局面评估服务。

优化说明：
  采用 MultiPV + root_moves 的底层接口，摒弃了“逐个评估子节点局面”的低效做法。
  一次请求仅调用一次 Stockfish 搜索，极大复用内部置换表和搜索树，
  性能相比逐个子节点评估提升 10 倍以上。

协议：4 字节 big-endian 长度头 + UTF-8 JSON 正文（支持同一连接上的多次请求）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import signal
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import chess
    import chess.engine
except ImportError:
    sys.exit("致命错误：缺少依赖，请执行 'pip install chess'")

# ──────────────────────────────────────────────────────────────────────────────
# 默认配置（可通过环境变量或命令行参数覆盖）
# ──────────────────────────────────────────────────────────────────────────────

STOCKFISH_PATH = os.getenv(
    "STOCKFISH_PATH",
    "/users/sgzwa126/stockfish/stockfish-ubuntu-x86-64-avx2",
)
SOCKET_PATH = os.getenv("SOCKET_PATH", "/tmp/stockfish_server.sock")
DEFAULT_DEPTH = int(os.getenv("DEFAULT_DEPTH", "18"))
NUM_ENGINES = int(os.getenv("NUM_ENGINES", "15"))
HASH_MB = int(os.getenv("HASH_MB", "256"))

# ──────────────────────────────────────────────────────────────────────────────
# 日志
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(module)s] %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 评分 → 胜率转换
# ──────────────────────────────────────────────────────────────────────────────


def score_to_win_prob_white(score: chess.engine.Score) -> float:
    """
    将 Stockfish 评分（白方视角）转为白方胜率。
    使用 Logistic 曲线 P = 1 / (1 + 10^(−cp/400))。
    """
    if score.is_mate():
        return 1.0 if score.mate() > 0 else 0.0
    cp = max(-3000, min(3000, score.score()))
    return 1.0 / (1.0 + math.pow(10.0, -cp / 400.0))


def extract_win_prob_white(info: chess.engine.InfoDict) -> float:
    """
    从 InfoDict 提取白方胜率（优先使用原生 WDL，没有则用 cp 转换）。
    """
    if "wdl" in info:
        wdl = info["wdl"].white()
        # 平局算 0.5，返回期望得分
        return (wdl.wins + wdl.draws * 0.5) / 1000.0

    if "score" in info:
        return score_to_win_prob_white(info["score"].white())

    # 极少数情况：引擎被强制中止未返回分数
    return 0.5


# ──────────────────────────────────────────────────────────────────────────────
# 引擎池
# ──────────────────────────────────────────────────────────────────────────────


class EnginePool:
    """
    基于 asyncio 的 Stockfish 进程池。
    采用借用/归还机制，支持 MultiPV 请求的并发处理。
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue
        self._handles: List[Tuple[Any, chess.engine.UciProtocol]] = []

    async def start(self, path: str, size: int, hash_mb: int) -> None:
        self._queue = asyncio.Queue()
        log.info("正在启动 %d 个 Stockfish 进程…", size)
        for i in range(size):
            transport, engine = await chess.engine.popen_uci(path)
            await engine.configure(
                {
                    "Threads": 1,
                    "Hash": hash_mb,
                    "UCI_ShowWDL": True,
                }
            )
            self._handles.append((transport, engine))
            await self._queue.put(engine)
            if (i + 1) % 5 == 0 or (i + 1) == size:
                log.info("  => 已就绪: %d / %d", i + 1, size)
        log.info("✅ 所有引擎启动完成并握手成功。")

    async def analyse_multipv(
        self,
        board: chess.Board,
        depth: int,
        multipv: int,
        root_moves: Optional[List[chess.Move]] = None,
    ) -> List[chess.engine.InfoDict]:
        """借用一个引擎、执行 MultiPV 分析、归还引擎。"""
        engine: chess.engine.UciProtocol = await self._queue.get()
        try:
            # 修正处：python-chess 中限制根节点着法的参数名为 root_moves 且放在 analyse 中
            limit = chess.engine.Limit(depth=depth)
            res = await engine.analyse(
                board, limit, multipv=multipv, root_moves=root_moves
            )
            return res if isinstance(res, list) else [res]
        finally:
            self._queue.put_nowait(engine)

    async def stop(self) -> None:
        log.info("正在优雅关闭引擎池…")
        for _, engine in self._handles:
            try:
                await asyncio.wait_for(engine.quit(), timeout=3.0)
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# 请求处理核心逻辑
# ──────────────────────────────────────────────────────────────────────────────


async def process_single(pool: EnginePool, req: dict) -> dict:
    """
    处理单个局面评估请求 (MultiPV 版本)。
    """
    fen = req.get("fen")
    if not fen:
        return {"error": "缺少必填字段：'fen'"}

    depth: int = max(1, min(int(req.get("depth", DEFAULT_DEPTH)), 40))
    filter_moves: Optional[List[str]] = req.get("moves")

    try:
        board = chess.Board(fen)
    except ValueError as exc:
        return {"error": f"无效的 FEN：{exc}"}

    turn_str = "white" if board.turn == chess.WHITE else "black"
    is_white = board.turn == chess.WHITE

    if board.is_game_over():
        outcome = board.outcome()
        wp_w = (
            0.5
            if outcome.winner is None
            else (1.0 if outcome.winner == chess.WHITE else 0.0)
        )
        wp_cur = wp_w if is_white else 1.0 - wp_w
        return {
            "fen": fen,
            "turn": turn_str,
            "game_over": True,
            "result": outcome.result(),
            "termination": outcome.termination.name,
            "current_win_prob": round(wp_cur, 6),
            "current_win_prob_white": round(wp_w, 6),
            "moves": [],
        }

    legal_moves: List[chess.Move] = list(board.legal_moves)
    if filter_moves is not None:
        wanted_set = set(filter_moves)
        root_moves = [m for m in legal_moves if m.uci() in wanted_set]
        if not root_moves:
            return {"error": "指定的 'moves' 中没有当前局面下的合法着法。"}
    else:
        root_moves = legal_moves

    num_multipv = len(root_moves)

    infos = await pool.analyse_multipv(
        board=board, depth=depth, multipv=num_multipv, root_moves=root_moves
    )

    if not infos:
        return {"error": "引擎内部错误，未返回任何评估结果。"}

    best_info = infos[0]
    wp_cur_w: float = extract_win_prob_white(best_info)
    wp_cur: float = wp_cur_w if is_white else 1.0 - wp_cur_w  # V(s)

    moves_out: List[Dict[str, Any]] = []
    seen_moves = set()

    for info in infos:
        if "pv" not in info or not info["pv"]:
            continue

        move = info["pv"][0]
        if move in seen_moves or move not in root_moves:
            continue
        seen_moves.add(move)

        wp_w = extract_win_prob_white(info)
        wp = wp_w if is_white else 1.0 - wp_w  # Q(s,a)
        delta = wp - wp_cur  # A(s,a) = Q(s,a) - V(s)

        b = board.copy(stack=False)
        b.push(move)

        moves_out.append(
            {
                "move": move.uci(),
                "san": board.san(move),
                "fen_after": b.fen(),
                "win_prob": round(wp, 6),
                "win_prob_delta": round(delta, 6),
                "win_prob_white": round(wp_w, 6),
            }
        )

    missing_moves = set(root_moves) - seen_moves
    for move in missing_moves:
        wp_w = 0.0 if is_white else 1.0
        wp = 0.0
        delta = wp - wp_cur

        b = board.copy(stack=False)
        b.push(move)

        moves_out.append(
            {
                "move": move.uci(),
                "san": board.san(move),
                "fen_after": b.fen(),
                "win_prob": round(wp, 6),
                "win_prob_delta": round(delta, 6),
                "win_prob_white": round(wp_w, 6),
            }
        )

    moves_out.sort(key=lambda m: m["win_prob"], reverse=True)

    return {
        "fen": fen,
        "turn": turn_str,
        "game_over": False,
        "current_win_prob": round(wp_cur, 6),
        "current_win_prob_white": round(wp_cur_w, 6),
        "depth": depth,
        "num_moves": len(moves_out),
        "moves": moves_out,
    }


async def process_request(pool: EnginePool, req: dict) -> dict:
    if "batch" in req:
        tasks = [process_single(pool, item) for item in req["batch"]]
        results = await asyncio.gather(*tasks)
        return {"results": list(results)}
    return await process_single(pool, req)


# ──────────────────────────────────────────────────────────────────────────────
# 通信协议
# ──────────────────────────────────────────────────────────────────────────────


async def recv_msg(reader: asyncio.StreamReader) -> Optional[dict]:
    try:
        hdr = await reader.readexactly(4)
    except asyncio.IncompleteReadError:
        return None
    size = int.from_bytes(hdr, "big")
    if size == 0:
        return None
    raw = await reader.readexactly(size)
    return json.loads(raw.decode("utf-8"))


async def send_msg(writer: asyncio.StreamWriter, obj: dict) -> None:
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    writer.write(len(raw).to_bytes(4, "big") + raw)
    await writer.drain()


# ──────────────────────────────────────────────────────────────────────────────
# 客户端连接处理
# ──────────────────────────────────────────────────────────────────────────────


async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    pool: EnginePool,
) -> None:
    peer = writer.get_extra_info("peername", "<unix_client>")
    log.info("🟢 客户端已连接：%s", peer)
    try:
        while True:
            req = await recv_msg(reader)
            if req is None:
                break
            try:
                resp = await process_request(pool, req)
            except Exception as exc:
                log.exception("❌ 处理请求时出现严重异常")
                resp = {"error": f"内部服务器错误: {str(exc)}"}
            await send_msg(writer, resp)
    except asyncio.IncompleteReadError:
        pass
    except Exception as exc:
        log.error("⚠️ 连接被意外中断：%s", exc)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        log.info("🔴 客户端已断开：%s", peer)


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────


async def amain(args: argparse.Namespace) -> None:
    pool = EnginePool()
    await pool.start(args.stockfish, args.engines, args.hash)

    sock_path: str = args.socket
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    server = await asyncio.start_unix_server(
        lambda r, w: handle_client(r, w, pool),
        path=sock_path,
    )
    os.chmod(sock_path, 0o660)

    log.info(
        "🚀 服务器监听中: %s | Depth: %d | MultiPV 引擎池规模: %d (Hash: %d MB/线程)",
        sock_path,
        args.depth,
        args.engines,
        args.hash,
    )

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    loop.add_signal_handler(signal.SIGTERM, stop.set)
    loop.add_signal_handler(signal.SIGINT, stop.set)

    async with server:
        await stop.wait()

    log.info("接收到终止信号，开始平滑关闭组件…")
    server.close()
    await pool.stop()
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    log.info("再见！进程退出。")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stockfish UNIX Socket Server (MultiPV 极速 RL 版)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--stockfish",
        default=STOCKFISH_PATH,
        metavar="PATH",
        help="Stockfish 二进制文件的绝对路径",
    )
    p.add_argument(
        "--socket",
        default=SOCKET_PATH,
        metavar="PATH",
        help="UNIX domain socket 通信通道路径",
    )
    p.add_argument(
        "--depth",
        default=DEFAULT_DEPTH,
        type=int,
        metavar="N",
        help="默认搜索深度 (推荐强化学习训练设为 12~18)",
    )
    p.add_argument(
        "--engines",
        default=NUM_ENGINES,
        type=int,
        metavar="N",
        help="并发引擎进程数 (建议: CPU核心数 - 1)",
    )
    p.add_argument(
        "--hash",
        default=HASH_MB,
        type=int,
        metavar="MB",
        help="每个引擎分配的置换表内存空间(MB)",
    )
    p.add_argument("--verbose", action="store_true", help="启用详细的 DEBUG 日志")
    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
