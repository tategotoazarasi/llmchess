"""Pygame-based chess GUI."""

import logging
import os
import threading
from typing import Optional

import chess
import pygame

# HiDPI Fixes
os.environ["SDL_VIDEO_MINIMIZE_ON_FOCUS"] = "0"
os.environ["SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR"] = "0"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"

from agents import ChessAgent
from chess_game import ChessGame

logger = logging.getLogger(__name__)

# 完整修复了所有缺失的颜色配置
_C = {
    "bg": (18, 18, 30),
    "panel_bg": (25, 25, 42),
    "panel_border": (55, 55, 85),
    "light_sq": (240, 217, 181),
    "dark_sq": (181, 136, 99),
    "highlight_sel": (246, 246, 105, 140),
    "highlight_last": (205, 210, 106, 100),
    "highlight_check": (235, 67, 52, 130),
    "text": (230, 230, 240),
    "text_dim": (140, 140, 165),
    "text_accent": (120, 180, 255),
    "text_white": (255, 255, 250),
    "text_black": (40, 40, 48),
    "move_dot": (100, 200, 100, 140),
    "capture_ring": (200, 80, 60, 160),
    "status_bar": (30, 30, 50),
}

_PIECE_GLYPH = {
    (chess.KING, chess.WHITE): "♔",
    (chess.QUEEN, chess.WHITE): "♕",
    (chess.ROOK, chess.WHITE): "♖",
    (chess.BISHOP, chess.WHITE): "♗",
    (chess.KNIGHT, chess.WHITE): "♘",
    (chess.PAWN, chess.WHITE): "♙",
    (chess.KING, chess.BLACK): "♚",
    (chess.QUEEN, chess.BLACK): "♛",
    (chess.ROOK, chess.BLACK): "♜",
    (chess.BISHOP, chess.BLACK): "♝",
    (chess.KNIGHT, chess.BLACK): "♞",
    (chess.PAWN, chess.BLACK): "♟",
}


class ChessGUI:
    SQ_SIZE = 80
    BOARD_PX = SQ_SIZE * 8
    PANEL_W = 320
    STATUS_H = 40
    COORD_MARGIN = 28

    def __init__(self, white: ChessAgent, black: ChessAgent):
        self.game = ChessGame(white, black)
        self._selected_sq: Optional[int] = None
        self._legal_targets: set[int] = set()
        self._last_move: Optional[chess.Move] = None
        self._ai_thinking = False
        self._ai_thread: Optional[threading.Thread] = None
        self._ai_move: Optional[chess.Move] = None
        self._status_msg = ""
        self.flip_board = False

        pygame.init()
        total_w = self.COORD_MARGIN + self.BOARD_PX + self.PANEL_W
        total_h = self.COORD_MARGIN + self.BOARD_PX + self.STATUS_H
        self.screen = pygame.display.set_mode(
            (total_w, total_h), pygame.SCALED | pygame.RESIZABLE
        )
        pygame.display.set_caption("LLM Chess — RL Grandmaster Edition")

        self._font_piece = pygame.font.SysFont("DejaVu Sans", self.SQ_SIZE - 12)
        self._font_coord = pygame.font.SysFont("DejaVu Sans", 14)
        self._font_ui = pygame.font.SysFont("Inter,Roboto,DejaVu Sans", 16)
        self._font_ui_bold = pygame.font.SysFont(
            "Inter,Roboto,DejaVu Sans", 16, bold=True
        )
        self.clock = pygame.time.Clock()

    def _sq_to_pixel(self, sq: int) -> tuple[int, int]:
        file, rank = chess.square_file(sq), chess.square_rank(sq)
        if self.flip_board:
            col, row = 7 - file, rank
        else:
            col, row = file, 7 - rank
        return self.COORD_MARGIN + col * self.SQ_SIZE, row * self.SQ_SIZE

    def _pixel_to_sq(self, x: int, y: int) -> Optional[int]:
        bx, by = x - self.COORD_MARGIN, y
        if bx < 0 or bx >= self.BOARD_PX or by < 0 or by >= self.BOARD_PX:
            return None
        col, row = bx // self.SQ_SIZE, by // self.SQ_SIZE
        if self.flip_board:
            file, rank = 7 - col, row
        else:
            file, rank = col, 7 - row
        return chess.square(file, rank)

    def _draw_board(self) -> None:
        board = self.game.board
        for sq in chess.SQUARES:
            x, y = self._sq_to_pixel(sq)
            is_light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
            pygame.draw.rect(
                self.screen,
                _C["light_sq"] if is_light else _C["dark_sq"],
                (x, y, self.SQ_SIZE, self.SQ_SIZE),
            )

        if self._last_move:
            for sq in (self._last_move.from_square, self._last_move.to_square):
                x, y = self._sq_to_pixel(sq)
                surf = pygame.Surface((self.SQ_SIZE, self.SQ_SIZE), pygame.SRCALPHA)
                surf.fill(_C["highlight_last"])
                self.screen.blit(surf, (x, y))

        if board.is_check():
            king_sq = board.king(board.turn)
            if king_sq is not None:
                x, y = self._sq_to_pixel(king_sq)
                surf = pygame.Surface((self.SQ_SIZE, self.SQ_SIZE), pygame.SRCALPHA)
                surf.fill(_C["highlight_check"])
                self.screen.blit(surf, (x, y))

        if self._selected_sq is not None:
            x, y = self._sq_to_pixel(self._selected_sq)
            surf = pygame.Surface((self.SQ_SIZE, self.SQ_SIZE), pygame.SRCALPHA)
            surf.fill(_C["highlight_sel"])
            self.screen.blit(surf, (x, y))

        for sq in self._legal_targets:
            x, y = self._sq_to_pixel(sq)
            surf = pygame.Surface((self.SQ_SIZE, self.SQ_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(
                surf, _C["move_dot"], (self.SQ_SIZE // 2, self.SQ_SIZE // 2), 10
            )
            self.screen.blit(surf, (x, y))

        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                glyph = _PIECE_GLYPH[(piece.piece_type, piece.color)]
                px, py = self._sq_to_pixel(sq)
                ts = self._font_piece.render(
                    glyph,
                    True,
                    (
                        _C["text_black"]
                        if piece.color == chess.BLACK
                        else _C["text_white"]
                    ),
                )
                tr = ts.get_rect(
                    center=(px + self.SQ_SIZE // 2, py + self.SQ_SIZE // 2)
                )
                self.screen.blit(ts, tr)

    def _draw_panel(self) -> None:
        ox = self.COORD_MARGIN + self.BOARD_PX
        pw, ph = self.PANEL_W, self.BOARD_PX + self.COORD_MARGIN
        pygame.draw.rect(self.screen, _C["panel_bg"], (ox, 0, pw, ph))
        pygame.draw.line(self.screen, _C["panel_border"], (ox, 0), (ox, ph), 2)

        y, pad = 20, 16
        ts = self._font_ui_bold.render("♚ Qwen RL Chess Arena", True, _C["text_accent"])
        self.screen.blit(ts, (ox + pad, y))
        y += 40

        for label, agent, color in [
            ("White", self.game.white, chess.WHITE),
            ("Black", self.game.black, chess.BLACK),
        ]:
            is_turn = self.game.board.turn == color and not self.game.is_over
            name_color = _C["text_accent"] if is_turn else _C["text"]
            ts = self._font_ui_bold.render(
                f"{label}: {agent.name}{' ⏳' if is_turn else ''}", True, name_color
            )
            self.screen.blit(ts, (ox + pad, y))
            y += 30

        y += 20
        if self.game.is_over:
            ts = self._font_ui_bold.render(
                f"Result: {self.game.result_str} ({self.game.result_reason})",
                True,
                (255, 200, 80),
            )
            self.screen.blit(ts, (ox + pad, y))
        elif self._ai_thinking:
            ts = self._font_ui.render(
                "AI is contemplating its next move...", True, _C["text_accent"]
            )
            self.screen.blit(ts, (ox + pad, y))

    def _draw_status_bar(self) -> None:
        y = self.BOARD_PX + self.COORD_MARGIN
        pygame.draw.rect(
            self.screen,
            _C["status_bar"],
            (0, y, self.screen.get_width(), self.STATUS_H),
        )
        ts = self._font_ui.render(
            self._status_msg or "Select a piece to move.", True, _C["text_dim"]
        )
        self.screen.blit(ts, (12, y + 12))

    def _handle_click(self, x: int, y: int) -> None:
        if (
            not self.game.current_agent.is_human
            or self.game.is_over
            or self._ai_thinking
        ):
            return
        sq = self._pixel_to_sq(x, y)
        if sq is None:
            self._selected_sq = None
            self._legal_targets.clear()
            return

        if self._selected_sq is not None and sq in self._legal_targets:
            move = chess.Move(self._selected_sq, sq)
            # Auto Queen Promotion
            if self.game.board.piece_at(
                self._selected_sq
            ).piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7):
                move = chess.Move(self._selected_sq, sq, promotion=chess.QUEEN)

            self._last_move = move
            self.game.push_move(move, agent=self.game.current_agent)
            self._selected_sq = None
            self._legal_targets.clear()
            return

        piece = self.game.board.piece_at(sq)
        if piece and piece.color == self.game.board.turn:
            self._selected_sq = sq
            self._legal_targets = {
                m.to_square for m in self.game.board.legal_moves if m.from_square == sq
            }
        else:
            self._selected_sq = None
            self._legal_targets.clear()

    def run(self) -> None:
        self.game.reset()
        self.game.start()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(*event.pos)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                    self.flip_board = not self.flip_board

            # Background AI processing
            if (
                not self.game.is_over
                and not self.game.current_agent.is_human
                and not self._ai_thinking
            ):
                self._ai_thinking = True
                board_copy = self.game.board.copy()
                agent = self.game.current_agent

                def _worker():
                    self._ai_move = agent.get_move(board_copy)

                self._ai_thread = threading.Thread(target=_worker, daemon=True)
                self._ai_thread.start()

            if self._ai_thinking and self._ai_thread and not self._ai_thread.is_alive():
                if self._ai_move and self._ai_move in self.game.board.legal_moves:
                    self._last_move = self._ai_move
                    self.game.push_move(self._ai_move, agent=self.game.current_agent)
                self._ai_thinking = False

            self.screen.fill(_C["bg"])
            self._draw_board()
            self._draw_panel()
            self._draw_status_bar()
            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()
