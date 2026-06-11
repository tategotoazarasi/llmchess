#!/usr/bin/env python3
"""LLM Chess — Play chess against the RL Qwen Model.

Usage examples:
  python main.py --white human --black llm
  python main.py --white llm --black stockfish
"""

import argparse
import logging
import sys

from agents import HumanAgent, LLMAgent, RandomAgent, StockfishAgent
from chess_gui import ChessGUI


def build_agent(kind: str, name: str, args) -> "ChessAgent":
    kind = kind.lower()
    if kind == "human":
        return HumanAgent(name=name)
    elif kind == "random":
        return RandomAgent(name=name)
    elif kind == "llm":
        return LLMAgent(api_base=args.url, model=args.model, name=name, temperature=0.0)
    elif kind == "stockfish":
        return StockfishAgent(path=args.sf_path, name=name, skill_level=args.sf_skill)
    else:
        sys.exit(f"Unknown agent type: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Chess Arena")
    parser.add_argument(
        "--white", choices=["human", "llm", "random", "stockfish"], default="human"
    )
    parser.add_argument(
        "--black", choices=["human", "llm", "random", "stockfish"], default="llm"
    )

    # 将 SGLang Endpoint 默认端口改为 8001
    parser.add_argument(
        "--url", default="http://localhost:8001/v1", help="API URL for LLM"
    )
    parser.add_argument("--model", default="chess-engine", help="Model name in SGLang")

    parser.add_argument(
        "--sf-path", default="stockfish", help="Path to Stockfish binary"
    )
    parser.add_argument(
        "--sf-skill", type=int, default=20, help="Stockfish skill level"
    )
    parser.add_argument(
        "--flip", action="store_true", help="Start with black at bottom"
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    white = build_agent(args.white, args.white.capitalize(), args)
    black = build_agent(
        args.black,
        "Qwen-RL (Black)" if args.black == "llm" else args.black.capitalize(),
        args,
    )

    gui = ChessGUI(white, black)
    gui.flip_board = args.flip
    gui.run()


if __name__ == "__main__":
    main()
