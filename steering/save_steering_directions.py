#!/usr/bin/env python3
"""
Precompute and save the steering-direction unit vector + typical_norm for
every model/size used in the steering experiments, so consumers of the repo
don't have to reload the dense model and redo the forward passes just to get
the direction (only needed if they want to *rebuild* it from scratch, e.g.
with different rollouts).

Saves, per model, a directory containing:
  - direction.npy   (float32 unit vector, hidden_size,)
  - meta.json       (model name, peak layer, typical_norm, source rollouts path, n_correct/n_wrong used)
"""
import gc
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = "/mnt/data/lannth/COMP/RAC/RAC/open-r1-main"  # cluster-specific -- point at your own rollouts location
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "directions")
MAX_SEQ_LEN = 4096

CONFIGS = [
    {
        "tag": "1.5B",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "rollouts_path": f"{PROJECT}/models/dense_1.5B_calib_for_steering/all_rollouts.json",
        "peak_layer": 15,
    },
    {
        "tag": "7B",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "rollouts_path": f"{PROJECT}/models/dense_7B_calib_for_steering/all_rollouts.json",
        "peak_layer": 15,
    },
    {
        "tag": "llama8B",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "rollouts_path": f"{PROJECT}/models/llama8B_daoc_sparse04/all_rollouts.json",
        "peak_layer": 12,
    },
    {
        "tag": "14B",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        # smaller/exploratory calibration set (12 problems, 80 rollouts) reused as-is rather than
        # regenerated -- see steering/README.md for why
        "rollouts_path": f"{PROJECT}/models/smoke_test_14B/all_rollouts.json",
        "peak_layer": 17,
    },
]


def compute_direction(model_name, rollouts_path, peak_layer, tag):
    print(f"[{tag}] Loading {model_name} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    device = next(model.parameters()).device

    with open(rollouts_path) as f:
        data = json.load(f)

    correct_vecs, wrong_vecs = [], []
    for prob in data:
        c, w = prob.get("correct", []), prob.get("wrong", [])
        if not c or not w:
            continue
        for rollouts, bucket in [(c[:6], correct_vecs), (w[:6], wrong_vecs)]:
            for r in rollouts:
                text = r.get("text", "") if isinstance(r, dict) else str(r)
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
                input_ids = enc["input_ids"].to(device)
                with torch.no_grad():
                    out = model(input_ids, output_hidden_states=True)
                seq_len = input_ids.shape[1]
                start = int(seq_len * 0.75)
                vec = out.hidden_states[peak_layer][0, start:].mean(dim=0).float().cpu().numpy()
                bucket.append(vec)
                del out

    correct_vecs = np.array(correct_vecs)
    wrong_vecs = np.array(wrong_vecs)
    direction = wrong_vecs.mean(axis=0) - correct_vecs.mean(axis=0)
    typical_norm = float(np.linalg.norm(np.concatenate([correct_vecs, wrong_vecs]), axis=1).mean())
    unit_direction = (direction / (np.linalg.norm(direction) + 1e-8)).astype(np.float32)

    out_dir = f"{OUT_DIR}/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/direction.npy", unit_direction)
    meta = {
        "model_name": model_name,
        "peak_layer": peak_layer,
        "typical_norm": typical_norm,
        "hidden_size": int(unit_direction.shape[0]),
        "source_rollouts_path": rollouts_path,
        "n_correct_rollouts": int(len(correct_vecs)),
        "n_wrong_rollouts": int(len(wrong_vecs)),
    }
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{tag}] Saved direction.npy + meta.json to {out_dir} | typical_norm={typical_norm:.3f} "
          f"n_correct={len(correct_vecs)} n_wrong={len(wrong_vecs)}", flush=True)

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()


for cfg in CONFIGS:
    out_dir = f"{OUT_DIR}/{cfg['tag']}"
    if os.path.exists(f"{out_dir}/direction.npy"):
        print(f"[{cfg['tag']}] already exists, skipping")
        continue
    compute_direction(cfg["model_name"], cfg["rollouts_path"], cfg["peak_layer"], cfg["tag"])

print("\n===== ALL DIRECTIONS SAVED =====")
