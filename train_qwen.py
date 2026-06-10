#!/usr/bin/env python3
"""
Chess LLM Trainer (SFT + GRPO Reinforcement Learning)
======================================================
"""

import sys

# ==============================================================================
# ⚠️ Unsloth 必须在首位导入
# ==============================================================================
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
except ImportError:
    sys.exit("致命错误：缺少依赖，请执行 'pip install unsloth'")

from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig

import json
import logging
import os
import re
from typing import List

import torch
import torch.distributed as dist
from datasets import load_dataset, Dataset

# ==============================================================================
# 🚀 核心修复：给 DDP 外壳打补丁，修复 Unsloth 双卡 GRPO 的 config 找不到的 Bug
# ==============================================================================
from torch.nn.parallel import DistributedDataParallel as DDP

if not hasattr(DDP, "config"):
    DDP.config = property(lambda self: self.module.config)
if not hasattr(DDP, "generation_config"):
    DDP.generation_config = property(lambda self: self.module.generation_config)


# ----------------- 分布式进程同步工具 -----------------
def is_main_process() -> bool:
    """判断当前是否为主进程 (Rank 0)"""
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def wait_for_everyone():
    """让所有 GPU 进程在此处停下等待，直到主进程完成文件读写"""
    if dist.is_initialized():
        dist.barrier()


# ==============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# 配置参数
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "unsloth/Qwen3.5-0.8B-Base"
MAX_SEQ_LENGTH = 1024
DATA_DIR = "/users/sgzwa126/llmchess/data"
SFT_DATA_PATH = os.path.join(DATA_DIR, "sft_data.jsonl")
RL_REWARD_PATH = os.path.join(DATA_DIR, "rl_rewards.json")

OUTPUT_DIR_SFT = "./chess_model_sft"
OUTPUT_DIR_RL = "./chess_model_rl_final"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 1. 奖励函数定义 (用于 GRPO)
# ──────────────────────────────────────────────────────────────────────────────

log.info("正在加载 RL 奖励字典...")
with open(RL_REWARD_PATH, "r", encoding="utf-8") as f:
    REWARD_DICT = json.load(f)


def extract_fen_from_prompt(prompt: str) -> str:
    match = re.search(r"FEN:\n(.*?)\nAnalyze", prompt)
    if match:
        return match.group(1).strip()
    return ""


def extract_move_from_completion(completion: str) -> str:
    match = re.search(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", completion.lower())
    if match:
        return match.group(1)
    return ""


def chess_reward_func(
    prompts: List[str], completions: List[str], **kwargs
) -> List[float]:
    rewards = []
    for prompt, completion_list in zip(prompts, completions):
        completion_text = (
            completion_list[0]["content"]
            if isinstance(completion_list, list)
            else completion_list
        )
        fen = extract_fen_from_prompt(prompt)
        move = extract_move_from_completion(completion_text)

        if not move or fen not in REWARD_DICT:
            rewards.append(-2.0)
            continue

        action_dict = REWARD_DICT[fen]
        if move not in action_dict:
            rewards.append(-1.0)
        else:
            delta = action_dict[move]
            # Advantage 增强
            rewards.append(delta * 10.0)

    return rewards


# ──────────────────────────────────────────────────────────────────────────────
# 核心训练流水线
# ──────────────────────────────────────────────────────────────────────────────


def build_chatml_prompt(fen: str) -> str:
    return (
        f"You are a Grandmaster chess engine. Given the current board state in FEN:\n"
        f"{fen}\n"
        f"Analyze the board and output the absolute best move in standard UCI format. "
        f"Output ONLY the UCI string (e.g., e2e4)."
    )


def main():
    log.info("🔥 启动 Unsloth 模型加载...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    # =====================================================================
    # 阶段一：Supervised Fine-Tuning (SFT)
    # =====================================================================
    log.info("🚀 阶段 1: 开始 SFT 训练...")
    sft_dataset = load_dataset("json", data_files={"train": SFT_DATA_PATH})["train"]

    def format_sft_dataset(examples):
        texts = [
            tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=False
            )
            for msg in examples["messages"]
        ]
        return {"text": texts}

    sft_dataset = sft_dataset.map(format_sft_dataset, batched=True)

    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=sft_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=4,
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=32,  # 🔥 极致榨干显存，双卡等效 64
            gradient_accumulation_steps=1,  # 🔥 禁用累积，速度起飞
            warmup_steps=100,
            max_steps=1500,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=OUTPUT_DIR_SFT,
            report_to="none",
            save_strategy="no",
        ),
    )
    sft_trainer.train()

    wait_for_everyone()
    if is_main_process():
        log.info("💾 SFT 完成，主进程保存中间模型...")
        model.save_pretrained_merged(OUTPUT_DIR_SFT, tokenizer, save_method="lora")
    wait_for_everyone()

    # =====================================================================
    # 阶段二：GRPO Reinforcement Learning
    # =====================================================================
    log.info("🚀 阶段 2: 开始 GRPO 强化学习...")

    import random

    all_fens = list(REWARD_DICT.keys())
    # 🔥 核心修复：必须全局打乱！让模型在一个 batch 里同时学到开局、中局和残局
    random.seed(42)
    random.shuffle(all_fens)

    prompts = []
    # 直接加载全部打乱后的 FEN（反正后面会被 max_steps 强制截停）
    for fen in all_fens:
        msgs = [
            {"role": "system", "content": "You are a highly capable AI chess player."},
            {"role": "user", "content": build_chatml_prompt(fen)},
        ]
        prompt_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt_text)

    rl_dataset = Dataset.from_dict({"prompt": prompts})

    grpo_trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[chess_reward_func],
        train_dataset=rl_dataset,
        args=GRPOConfig(
            output_dir=OUTPUT_DIR_RL,
            per_device_train_batch_size=8,  # 🔥 双卡共 32 个 prompts。每步同时生成 32x8=256 条文本！
            gradient_accumulation_steps=2,  # 🔥 禁用累积，每走一步立马更新梯度，速度十倍提升
            learning_rate=1e-5,
            num_generations=8,
            max_prompt_length=256,
            max_completion_length=32,
            max_steps=600,  # 🔥 因为速度很快，增加 RL 步数让它下得更好！
            save_strategy="no",
            logging_steps=10,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            optim="adamw_8bit",
            report_to="none",
        ),
    )

    grpo_trainer.train()

    # ---------------- DDP 安全保存机制 (RL) ----------------
    wait_for_everyone()  # 让所有进程停下
    if is_main_process():
        log.info("💾 强化学习完成，主进程开始保存最终 16-bit 模型...")
        model.save_pretrained_merged(
            OUTPUT_DIR_RL, tokenizer, save_method="merged_16bit"
        )
        log.info("🎉 全部训练任务圆满完成！模型已安全落地！")
    wait_for_everyone()
    # -------------------------------------------------------


if __name__ == "__main__":
    main()
