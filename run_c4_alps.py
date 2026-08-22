#!/usr/bin/env python3
"""
C4 Calibration Baseline for LRM Pruning — ALPS backend.

Same recipe as run_c4_baseline.py (SparseGPT+C4) but pruning with ALPS
(ADMM sparse least-squares) instead of SparseGPT. Serves as the ALPS-backend
counterpart to the C4 baseline, for comparison against RAC/OBC-Prune (ALPS).

Usage:
    python run_c4_alps.py \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --sparsity 0.4 \
        --output_dir ./models/c4_alps_15B_sparse40
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the existing TRL ALPS infrastructure
from trl.pruner.pruning import alps_prune, compute_sparsity
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("c4_alps")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C4 calibration ALPS pruning baseline")
    p.add_argument("--model_name", required=True,
                   help="HF model name, e.g. deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--sparsity", type=float, required=True,
                   help="Target sparsity, e.g. 0.4 for 40%")
    p.add_argument("--output_dir", required=True,
                   help="Directory to save the pruned model")
    p.add_argument("--n_calib_samples", type=int, default=128,
                   help="Number of C4 calibration sequences (default: 128)")
    p.add_argument("--calib_seq_len", type=int, default=2048,
                   help="Length of each calibration sequence in tokens (default: 2048)")
    p.add_argument("--device", default="auto",
                   help="Device map for model loading (default: auto)")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float16", "bfloat32", "bfloat16"],
                   help="Model dtype (default: bfloat16)")
    p.add_argument("--scope", default="all", choices=["all", "mlp"],
                   help="Which layers to prune (default: all)")
    p.add_argument("--memory_limit_gb", type=float, default=100.0,
                   help="GPU memory budget for XtX groups (default: 100 GB)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for C4 sampling (default: 42)")
    p.add_argument("--alps_rho", type=float, default=0.1,
                   help="ALPS ADMM penalty parameter (default: 0.1)")
    p.add_argument("--alps_max_iter", type=int, default=300,
                   help="ALPS ADMM max iterations (default: 300)")
    return p.parse_args()


def load_c4_calibration(
    tokenizer: AutoTokenizer,
    n_samples: int = 128,
    seq_len: int = 2048,
    seed: int = 42,
) -> DataLoader:
    """
    Load C4 calibration data: n_samples random text chunks of seq_len tokens each.
    Uses streaming to avoid downloading all of C4.
    """
    import random
    random.seed(seed)

    logger.info("Loading C4 dataset (en subset, streaming) ...")
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    shuffled = ds.shuffle(seed=seed, buffer_size=10000)

    samples = []
    total_tokens = 0
    target_tokens = n_samples * seq_len
    pad_id = tokenizer.pad_token_id or 0

    for example in shuffled:
        if len(samples) >= n_samples:
            break
        text = example["text"]
        enc = tokenizer.encode(text, add_special_tokens=False)
        if len(enc) < seq_len:
            # Pad short sequences
            enc = enc + [pad_id] * (seq_len - len(enc))
        else:
            # Take random contiguous chunk
            start = random.randint(0, len(enc) - seq_len)
            enc = enc[start:start + seq_len]
        samples.append(torch.tensor(enc, dtype=torch.long))
        total_tokens += seq_len
        if len(samples) % 32 == 0:
            logger.info(
                "Collected %d/%d C4 samples (~%d tokens)",
                len(samples), n_samples, total_tokens,
            )

    logger.info(
        "Collected %d C4 samples (~%d tokens, target %d)",
        len(samples), total_tokens, target_tokens,
    )

    class C4Dataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return {"input_ids": self.samples[idx]}

    def collate_fn(batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch],
            batch_first=True, padding_value=pad_id,
        )
        attention_mask = (input_ids != pad_id).long()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }

    dataset = C4Dataset(samples)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
    )
    return loader


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # ── 1. Load model + tokenizer ──────────────────────────────────────────
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    model.config.use_cache = False

    # ── 2. Load C4 calibration data ────────────────────────────────────────
    logger.info(
        f"Loading C4 calibration: {args.n_calib_samples} samples × "
        f"{args.calib_seq_len} tokens"
    )
    calib_loader = load_c4_calibration(
        tokenizer,
        n_samples=args.n_calib_samples,
        seq_len=args.calib_seq_len,
        seed=args.seed,
    )

    # ── 3. Prune ───────────────────────────────────────────────────────────
    logger.info(
        f"Starting ALPS pruning (C4 calibration) at {args.sparsity*100:.0f}% sparsity"
    )
    t0 = time.time()

    alps_prune(
        model,
        calib_loader,
        args.sparsity,
        device="cuda" if torch.cuda.is_available() else "cpu",
        scope=args.scope,
        memory_limit_gb=args.memory_limit_gb,
        rho=args.alps_rho,
        max_iter=args.alps_max_iter,
    )

    elapsed = time.time() - t0
    logger.info(f"Pruning done in {elapsed:.1f}s")

    # ── 4. Report sparsity ─────────────────────────────────────────────────
    realised = compute_sparsity(model)
    logger.info(f"Realised sparsity: {realised * 100:.2f}%")

    # ── 5. Save ────────────────────────────────────────────────────────────
    model.config.use_cache = True
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Saved pruned model to {output_dir}")

    # Write summary
    summary = {
        "condition": "c4_alps",
        "sparsity_target": args.sparsity,
        "sparsity_realised": realised,
        "n_calib_samples": args.n_calib_samples,
        "calib_seq_len": args.calib_seq_len,
        "model": args.model_name,
        "pruning_time_s": elapsed,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
