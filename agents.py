"""Chess agent implementations."""

import abc
import logging
import random
import re
import time
from typing import Optional

import chess
import chess.engine
import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)

timeout_config = httpx.Timeout(30.0, connect=5.0, read=25.0, write=5.0)


class ChessAgent(abc.ABC):
    def __init__(self, name: str):
        self.name = name
        self._color: Optional[chess.Color] = None
        self.last_move_info = {}

    @abc.abstractmethod
    def get_move(self, board: chess.Board, **kwargs) -> chess.Move: ...

    @property
    def is_human(self) -> bool:
        return False

    def on_game_start(self, color: chess.Color) -> None:
        self._color = color

    def on_game_end(self, result: str) -> None:
        pass


class RandomAgent(ChessAgent):
    def __init__(self, name: str = "Random"):
        super().__init__(name)

    def get_move(self, board: chess.Board, **kwargs) -> chess.Move:
        return random.choice(list(board.legal_moves))


class HumanAgent(ChessAgent):
    def __init__(self, name: str = "Human"):
        super().__init__(name)

    @property
    def is_human(self) -> bool:
        return True

    def get_move(self, board: chess.Board, **kwargs) -> chess.Move:
        legal_moves = list(board.legal_moves)
        while True:
            try:
                uci_str = input("Your move (UCI, e.g. e2e4): ").strip().lower()
                move = chess.Move.from_uci(uci_str)
                if move in legal_moves:
                    return move
                promo = chess.Move.from_uci(uci_str + "q")
                if promo in legal_moves:
                    return promo
                print("  ✗ Illegal move.")
            except ValueError:
                print("  ✗ Invalid UCI format.")


class LLMAgent(ChessAgent):
    """Agent powered by the custom RL Qwen model via SGLang."""

    def __init__(
        self,
        api_base: str = "http://localhost:8001/v1",
        api_key: str = "not-needed",
        model: str = "chess-engine",
        name: str = "Qwen-RL",
        temperature: float = 0.0,
    ):
        super().__init__(name)
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=timeout_config,
        )
        self.model = model
        self.temperature = temperature

    def get_move(self, board: chess.Board, **kwargs) -> chess.Move:
        legal_moves = list(board.legal_moves)
        legal_ucis = [m.uci() for m in legal_moves]

        system_msg = "You are a highly capable AI chess player."
        user_msg = (
            f"You are a Grandmaster chess engine. Given the current board state in FEN:\n"
            f"{board.fen()}\n"
            f"Analyze the board and output the absolute best move in standard UCI format. "
            f"Output ONLY the UCI string (e.g., e2e4)."
        )

        # 构造绝对合法的枚举约束（类似 JSON Schema 中的 Enum 机制）
        # SGLang/vLLM 支持使用 guided_choice 限制输出范围
        strict_regex = r"^(" + "|".join(legal_ucis) + r")$"

        # 加入温度退避重试机制 (Temperature Backoff)
        # 如果模型输出了幻觉，提高温度逼迫它输出其他可能性，而不是直接放弃并使用 random
        for attempt, temp in enumerate([self.temperature, 0.3, 0.6, 0.9]):
            try:
                t0 = time.perf_counter()

                # 传入所有可能的 FSM 约束参数，确保彻底击穿 SGLang 的 API 壁垒
                extra_args = {
                    "guided_choice": legal_ucis,
                    "guided_regex": strict_regex,
                    "regex": strict_regex,
                }

                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=temp,
                    max_tokens=8,
                    stream=True,
                    extra_body=extra_args,
                )

                attempt_str = f"(Temp={temp})" if attempt > 0 else "(Streaming)"
                print(
                    f"\n[{self.name}] 🧠 Thinking {attempt_str}: ", end="", flush=True
                )

                move_str = ""
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if getattr(delta, "content", None) is not None:
                            text = delta.content
                            move_str += text
                            print(text, end="", flush=True)

                elapsed = time.perf_counter() - t0
                print(f"  [⏱️ {elapsed:.3f}s]")

                move_str = move_str.strip()

                # 双重保险验证
                match = re.search(r"[a-h][1-8][a-h][1-8][qrbn]?", move_str.lower())
                if match:
                    extracted_uci = match.group(0)
                    move = chess.Move.from_uci(extracted_uci)
                    if move in legal_moves:
                        return move
                    else:
                        logger.warning(
                            "[%s] Generated illegal move: '%s'. Retrying...",
                            self.name,
                            extracted_uci,
                        )
                else:
                    logger.warning(
                        "[%s] Failed to extract valid UCI from '%s'. Retrying...",
                        self.name,
                        move_str,
                    )

            except Exception as e:
                logger.error("[%s] API Error: %s", self.name, e)

        # 只有当 4 次不同温度的重试全部失败（极度罕见），才会退回到随机
        fallback = random.choice(legal_moves)
        logger.error(
            "[%s] 🚨 ALL constraints and retries failed! Falling back to random move: %s",
            self.name,
            fallback.uci(),
        )
        return fallback


class StockfishAgent(ChessAgent):
    def __init__(
        self,
        path: str = "stockfish",
        name: str = "Stockfish",
        skill_level: int = 20,
        time_limit: float = 0.5,
    ):
        super().__init__(name)
        self.path = path
        self.skill_level = skill_level
        self.time_limit = time_limit
        self._engine = None

    def _ensure_engine(self) -> None:
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
            self._engine.configure({"Skill Level": self.skill_level})

    def on_game_start(self, color: chess.Color) -> None:
        super().on_game_start(color)
        self._ensure_engine()

    def on_game_end(self, result: str) -> None:
        super().on_game_end(result)
        if self._engine:
            self._engine.quit()
            self._engine = None

    def get_move(self, board: chess.Board, **kwargs) -> chess.Move:
        self._ensure_engine()
        result = self._engine.play(board, chess.engine.Limit(time=self.time_limit))
        return result.move
