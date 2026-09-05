#!/usr/bin/env python3
"""
Cross-model variant of the loop-vs-correct direction: "correct" rollouts
come from the DENSE model's own rollouts (encoded via the dense model's
forward pass), "looped_wrong" rollouts come from the PRUNED model's own
rollouts (encoded via the pruned model's forward pass) -- each bucket
encoded by whichever model actually generated it, rather than both encoded
by a single model. Tests whether pushing the pruned model's activations
toward where the DENSE model's genuinely-successful reasoning lives is a
more effective lever than contrasting against the pruned model's own
(much rarer) correct rollouts.

Usage:
    python build_cross_model_loop_direction.py <dense_model_path> <dense_rollouts_json> \
        <pruned_model_path> <pruned_rollouts_json> <peak_layer> <out_dir>
"""
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_SEQ_LEN = 4096


def encode_bucket(model, tokenizer, records, bucket_key, peak_layer, device):
    vecs = []
    for prob in records:
        for r in prob.get(bucket_key, []):
            text = r.get("text", "") if isinstance(r, dict) else str(r)
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
            input_ids = enc["input_ids"].to(device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True)
            seq_len = input_ids.shape[1]
            start = int(seq_len * 0.75)
            vec = out.hidden_states[peak_layer][0, start:].mean(dim=0).float().cpu().numpy()
            vecs.append(vec)
            del out
    return vecs


def main():
    dense_model_path = sys.argv[1]
    dense_rollouts_path = sys.argv[2]
    pruned_model_path = sys.argv[3]
    pruned_rollouts_path = sys.argv[4]
    peak_layer = int(sys.argv[5])
    out_dir = sys.argv[6]
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading DENSE {dense_model_path} ...", flush=True)
    dense_tokenizer = AutoTokenizer.from_pretrained(dense_model_path)
    dense_model = AutoModelForCausalLM.from_pretrained(dense_model_path, dtype=torch.bfloat16, device_map="auto")
    dense_model.eval()
    dense_device = next(dense_model.parameters()).device

    with open(dense_rollouts_path) as f:
        dense_records = json.load(f)
    correct_vecs = encode_bucket(dense_model, dense_tokenizer, dense_records, "correct", peak_layer, dense_device)
    n_correct = len(correct_vecs)
    print(f"Encoded {n_correct} DENSE 'correct' rollouts at layer {peak_layer}", flush=True)

    del dense_model, dense_tokenizer
    torch.cuda.empty_cache()

    print(f"Loading PRUNED {pruned_model_path} ...", flush=True)
    pruned_tokenizer = AutoTokenizer.from_pretrained(pruned_model_path)
    pruned_model = AutoModelForCausalLM.from_pretrained(pruned_model_path, dtype=torch.bfloat16, device_map="auto")
    pruned_model.eval()
    pruned_device = next(pruned_model.parameters()).device

    with open(pruned_rollouts_path) as f:
        pruned_records = json.load(f)
    looped_vecs = encode_bucket(pruned_model, pruned_tokenizer, pruned_records, "looped_wrong", peak_layer, pruned_device)
    n_looped = len(looped_vecs)
    print(f"Encoded {n_looped} PRUNED 'looped_wrong' rollouts at layer {peak_layer}", flush=True)

    if not correct_vecs or not looped_vecs:
        raise RuntimeError(f"Empty bucket: {n_correct} correct, {n_looped} looped -- cannot build a direction.")

    correct_vecs, looped_vecs = np.array(correct_vecs), np.array(looped_vecs)
    direction = looped_vecs.mean(axis=0) - correct_vecs.mean(axis=0)
    typical_norm = float(np.linalg.norm(np.concatenate([correct_vecs, looped_vecs]), axis=1).mean())
    unit_direction = (direction / (np.linalg.norm(direction) + 1e-8)).astype(np.float32)

    np.save(f"{out_dir}/direction.npy", unit_direction)
    meta = {
        "dense_model_path": dense_model_path, "pruned_model_path": pruned_model_path,
        "peak_layer": peak_layer, "typical_norm": typical_norm, "hidden_size": int(unit_direction.shape[0]),
        "dense_rollouts_path": dense_rollouts_path, "pruned_rollouts_path": pruned_rollouts_path,
        "direction_semantics": "PRUNED model's looped_wrong (encoded by pruned model) minus "
                                "DENSE model's correct (encoded by dense model)",
        "n_correct_rollouts_used": n_correct, "n_looped_wrong_rollouts_used": n_looped,
    }
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved direction.npy + meta.json to {out_dir} | typical_norm={typical_norm:.3f} "
          f"n_correct={n_correct} n_looped={n_looped}", flush=True)


if __name__ == "__main__":
    main()
