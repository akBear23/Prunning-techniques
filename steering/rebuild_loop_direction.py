#!/usr/bin/env python3
"""
Standalone direction (re)builder: loads an already-saved all_rollouts.json
(from sample_pruned_loop_direction.py) and computes the loop-vs-correct
direction from it, without regenerating any rollouts. Use this to rebuild
with an updated build_direction() (e.g. after relaxing the per-question
pairing requirement) without re-running the expensive sampling step.

Usage:
    python rebuild_loop_direction.py <model_path> <peak_layer> <rollouts_json> <out_dir>
"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sample_pruned_loop_direction import build_direction, MAX_SEQ_LEN


def main():
    model_path = sys.argv[1]
    peak_layer = int(sys.argv[2])
    rollouts_path = sys.argv[3]
    out_dir = sys.argv[4]

    import os
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    device = next(model.parameters()).device

    with open(rollouts_path) as f:
        records = json.load(f)
    n_c = sum(len(p.get("correct", [])) for p in records)
    n_l = sum(len(p.get("looped_wrong", [])) for p in records)
    print(f"Loaded {len(records)} problems from {rollouts_path} (correct={n_c}, looped_wrong={n_l})", flush=True)

    unit_direction, typical_norm, n_correct_used, n_looped_used = build_direction(
        model, tokenizer, device, records, peak_layer
    )

    import numpy as np
    np.save(f"{out_dir}/direction.npy", unit_direction)
    meta = {
        "model_path": model_path, "peak_layer": peak_layer, "typical_norm": typical_norm,
        "hidden_size": int(unit_direction.shape[0]), "source_rollouts_path": rollouts_path,
        "direction_semantics": "looped_wrong (cannot stop) minus correct (terminates + right answer), "
                                "pooled globally across all problems (no per-question pairing required)",
        "n_correct_rollouts_used": n_correct_used, "n_looped_wrong_rollouts_used": n_looped_used,
    }
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved direction.npy + meta.json to {out_dir} | typical_norm={typical_norm:.3f} "
          f"n_correct_used={n_correct_used} n_looped_used={n_looped_used}", flush=True)


if __name__ == "__main__":
    main()
