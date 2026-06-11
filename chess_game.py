"""Chess game engine and state management."""

import logging
import time
from typing import Optional

import chess
import chess.engine

from agents import ChessAgent

logger = logging.getLogger(__name__)


class ChessGame:
    """Core chess game engine."""

    def __init__(self, white: ChessAgent, black: ChessAgent, max_moves: int = 200):
        self.white = white
        self.black = black
        self.max_moves = max_moves
        self.board = chess.Board()
        self.session_id = int(time.time())
        self._resigned_by: Optional[str] = None

        # Used for drawing eval graph if available
        self.eval_history = [0.0]

    @property
    def current_agent(self) -> ChessAgent:
        return self.white if self.board.turn == chess.WHITE else self.black

    @property
    def is_over(self) -> bool:
        return (
            self.board.is_game_over(claim_draw=True)
            or self.board.fullmove_number > self.max_moves
            or self._resigned_by is not None
        )

    @property
    def result_str(self) -> str:
        if self._resigned_by == "white":
            return "0-1"
        if self._resigned_by == "black":
            return "1-0"
        if self.board.is_checkmate():
            return "0-1" if self.board.turn == chess.WHITE else "1-0"
        return "1/2-1/2"

    @property
    def result_reason(self) -> str:
        if self._resigned_by:
            return "resignation"
        if self.board.is_checkmate():
            return "checkmate"
        if self.board.is_stalemate():
            return "stalemate"
        if self.board.is_insufficient_material():
            return "insufficient material"
        if self.board.can_claim_fifty_moves():
            return "fifty-move rule"
        if self.board.can_claim_threefold_repetition():
            return "threefold repetition"
        if self.board.fullmove_number > self.max_moves:
            return "move limit exceeded"
        return "unknown"

    def reset(self) -> None:
        self.board = chess.Board()
        self._resigned_by = None
        self.eval_history = [0.0]

    def start(self) -> None:
        self.white.on_game_start(chess.WHITE)
        self.black.on_game_start(chess.BLACK)

    def push_move(self, move: chess.Move, agent: ChessAgent = None) -> None:
        san = self.board.san(move)
        uci = move.uci()
        self.board.push(move)
        logger.info(
            "Move: %s played %s (%s)", agent.name if agent else "Unknown", san, uci
        )
