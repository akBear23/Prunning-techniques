#!/usr/bin/env python3
"""
C4 calibration with Wanda and ALPS backends.
Loads C4 text data, runs Wanda or ALPS pruning, saves model.

Usage:
    python run_c4_wanda_alps.py \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --sparsity 0.4 --backend wanda \
        --output_dir ./models/c4_wanda_1.5B_sparse40
"""
from __future__ import annotations

import argparse, json, logging, os, time, sys
import torch, torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("c4_wanda_alps")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--sparsity", type=float, required=True)
    p.add_argument("--backend", required=True, choices=["wanda", "alps"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_calib_samples", type=int, default=128)
    p.add_argument("--calib_seq_len", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--scope", default="all")
    p.add_argument("--memory_limit_gb", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alps_rho", type=float, default=0.1)
    p.add_argument("--alps_max_iter", type=int, default=300)
    return p.parse_args()


def load_c4_calibration(tokenizer, n_samples=128, seq_len=2048, seed=42):
    """Load C4 calibration data as a DataLoader."""
    logger.info(f"Loading C4 dataset (en, streaming) ...")
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    shuffled = ds.shuffle(seed=seed, buffer_size=10000)

    samples = []
    target_tokens = n_samples * seq_len
    for example in shuffled:
        text = example["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        samples.extend(tokens)
        if len(samples) >= target_tokens:
            break

    samples = samples[:target_tokens]
    logger.info(f"Collected {len(samples)} tokens total")

    chunks = [samples[i:i+seq_len] for i in range(0, len(samples)-seq_len+1, seq_len)]
    chunks = chunks[:n_samples]
    logger.info(f"Split into {len(chunks)} chunks of ~{seq_len} tokens")

    class C4Dataset(torch.utils.data.Dataset):
        def __init__(self, chunks):
            self.chunks = [torch.tensor(c, dtype=torch.long) for c in chunks]
        def __len__(self):
            return len(self.chunks)
        def __getitem__(self, idx):
            return self.chunks[idx]

    return DataLoader(C4Dataset(chunks), batch_size=1, shuffle=False)


def prune_wanda_c4(model, calib_loader, sparsity, *, device="cuda", scope="all"):
    """Standard Wanda pruning with C4 calibration (no per-token weights)."""
    from trl.pruner.pruning import WrappedGPT, _is_mlp

    PRUNE_TYPES = (nn.Linear,)
    layer_list = [
        (n, m) for n, m in model.named_modules()
        if isinstance(m, PRUNE_TYPES) and m.weight.requires_grad
        and (scope == "all" or _is_mlp(n))
    ]
    logger.info(f"[C4-Wanda] {len(layer_list)} layers selected for pruning")

    wrappers = {name: WrappedGPT(lyr) for name, lyr in layer_list}

    # Forward pass to accumulate scaler_row (no token weights)
    hooks = []
    for name, wr in wrappers.items():
        def _make_hook(wr):
            def _hook(mod, inp, out):
                wr.add_batch(inp[0].detach(), out.detach())
            return _hook
        mod = dict(layer_list)[name]
        hooks.append(mod.register_forward_hook(_make_hook(wr)))

    model = model.to(device)
    model.eval()
    detected_device = next(model.parameters()).device
    logger.info(f"Model on {detected_device}")
    with torch.no_grad():
        for batch_idx, batch in enumerate(calib_loader):
            input_ids = batch.to(detected_device)
            model(input_ids)
    for h in hooks:
        h.remove()

    # Prune using Wanda scores: |W_ij| * ||X_j||_2
    total_params, pruned_params = 0, 0
    for name, wr in wrappers.items():
        W = dict(layer_list)[name].weight.data
        scaler = wr.scaler_row.to(W.device).view(1, -1)
        # Wanda score
        scores = W.abs() * scaler.sqrt()
        for row_idx in range(scores.shape[0]):
            k = int(sparsity * scores.shape[1])
            if k <= 0:
                continue
            _, idx = torch.topk(scores[row_idx], k, largest=False)
            W[row_idx, idx] = 0.0
            pruned_params += k
        total_params += W.numel()

    actual = pruned_params / total_params if total_params > 0 else 0
    logger.info(f"[C4-Wanda] Pruned {pruned_params}/{total_params} ({actual:.2%} actual "
                f"vs {sparsity:.2%} target)")
    return model


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dtype = getattr(torch, args.dtype)
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, trust_remote_code=True,
    )
    model = model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    calib_loader = load_c4_calibration(tokenizer, args.n_calib_samples,
                                        args.calib_seq_len, args.seed)

    t0 = time.time()
    if args.backend == "wanda":
        prune_wanda_c4(model, calib_loader, args.sparsity,
                       device=args.device, scope=args.scope)
    elif args.backend == "alps":
        from trl.pruner.pruning import ALPS_prune
        if ALPS_prune is None:
            raise ImportError("ALPS_prune not available")
        logger.info(f"[C4-ALPS] Starting ALPS pruning")
        ALPS_prune(model, calib_loader, args.sparsity,
                   device=args.device, scope=args.scope,
                   memory_limit_gb=args.memory_limit_gb,
                   rho=args.alps_rho, max_iter=args.alps_max_iter)
    elapsed = time.time() - t0
    logger.info(f"Pruning completed in {elapsed/60:.1f} min")

    logger.info(f"Saving to {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    with open(f"{args.output_dir}/run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    logger.info("Done.")


if __name__ == "__main__":
    main()
