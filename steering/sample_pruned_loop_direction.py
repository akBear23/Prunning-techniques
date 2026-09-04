#!/usr/bin/env python3
"""
Build a steering direction from a PRUNED model's own rollouts, contrasting
"can finish and get the right answer" against "loops until it dies" -- a
different split than the project's usual correct/wrong-ANSWER direction:

  - correct:        terminates naturally (not truncated) AND matches gold.
  - looped_wrong:    truncated (hit max_new_tokens, or the repetition-loop
                     detector cut it short) -- the "cannot stop" failure
                     mode this project has been characterizing all session
                     (verbatim repetition loops account for ~90-99% of
                     these at the 16384-token cap).
  - terminated_wrong: terminates naturally but the final answer is wrong --
                     saved for completeness, but NOT used to build the
                     direction (neither "succeeds" nor "loops").

direction = mean(looped_wrong hidden states) - mean(correct hidden states),
same last-quarter-mean-at-peak-layer methodology used throughout this
project's steering work.

Usage:
    python sample_pruned_loop_direction.py <model_path> <peak_layer> <out_dir> \
        [n_problems] [n_rollouts_per_problem]
"""
import json
import os
import sys

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList

from lighteval.metrics.utils.extractive_match_utils import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    get_extraction_regexes,
    extract_target_from_pred,
)
from lighteval.metrics.utils.math_comparison import compare_gold_target
from lighteval.utils.language import Language

from repetition_stopping import RepetitionStoppingCriteria

MAX_SEQ_LEN = 4096
MAX_NEW_TOKENS = 16384
TEMPERATURE = 0.8

MATH_QUERY_TEMPLATE = """
Solve the following problem. The final line of your response MUST be of the following format:
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the final answer. Think step by step before answering.

{prompt}
""".strip()

PRED_REGEXES = get_extraction_regexes(None, [ExprExtractionConfig(), LatexExtractionConfig()], Language.ENGLISH)
GOLD_REGEXES = get_extraction_regexes(None, [ExprExtractionConfig(), LatexExtractionConfig()], Language.ENGLISH)


def is_correct(pred_text, gold_text):
    ep = extract_target_from_pred(pred_text, PRED_REGEXES, "first_match", "any_match", 5)
    eg = extract_target_from_pred(gold_text, GOLD_REGEXES, "first_match", "any_match", 5)
    if not ep or not eg:
        return False
    try:
        return bool(compare_gold_target(eg, ep, precision=6))
    except Exception:
        return False


def gold_extractable(gold_text):
    return bool(extract_target_from_pred(gold_text, GOLD_REGEXES, "first_match", "any_match", 5))


def generate_rollouts(model, tokenizer, device, n_problems, n_rollouts):
    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train")
    candidates, candidate_indices = [], []
    for i in range(len(ds)):
        if len(candidates) >= n_problems:
            break
        row = ds[i]
        gold = f"ANSWER: {row['answer']}"
        if gold_extractable(gold):
            candidates.append(row)
            candidate_indices.append(i)
    print(f"Selected {len(candidates)} problems with extractable gold answers", flush=True)

    out_records = []
    for pi, (row, ds_idx) in enumerate(zip(candidates, candidate_indices)):
        query = MATH_QUERY_TEMPLATE.format(prompt=row["problem"])
        messages = [{"role": "user", "content": query}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
        input_ids = enc["input_ids"].to(device)
        prompt_len = input_ids.shape[1]
        rep_criteria = RepetitionStoppingCriteria(tokenizer, prompt_len)
        with torch.no_grad():
            gen = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                num_return_sequences=n_rollouts,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([rep_criteria]),
            )
        gold = f"ANSWER: {row['answer']}"
        correct_rollouts, looped_wrong_rollouts, terminated_wrong_rollouts = [], [], []
        for r in range(gen.shape[0]):
            new_tokens = gen[r, prompt_len:]
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            n_nonpad = (new_tokens != pad_id).sum().item()
            repetition_stopped = bool(rep_criteria.stopped and rep_criteria.stopped[r])
            truncated = n_nonpad >= MAX_NEW_TOKENS or repetition_stopped
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
            full_text = prompt_text + completion
            correct = is_correct(completion, gold)
            record = {"text": full_text, "token_count": int(n_nonpad),
                      "truncated": bool(truncated), "repetition_stopped": repetition_stopped,
                      "final_correct": bool(correct)}
            if truncated:
                looped_wrong_rollouts.append(record)
            elif correct:
                correct_rollouts.append(record)
            else:
                terminated_wrong_rollouts.append(record)
        out_records.append({
            "problem_id": ds_idx, "problem": row["problem"],
            "correct": correct_rollouts,
            "looped_wrong": looped_wrong_rollouts,
            "terminated_wrong": terminated_wrong_rollouts,
        })
        n_c = sum(len(p["correct"]) for p in out_records)
        n_l = sum(len(p["looped_wrong"]) for p in out_records)
        n_t = sum(len(p["terminated_wrong"]) for p in out_records)
        print(f"  {pi+1}/{len(candidates)} problems done "
              f"(correct={n_c}, looped_wrong={n_l}, terminated_wrong={n_t} so far)", flush=True)
        del gen
        torch.cuda.empty_cache()

    return out_records


def build_direction(model, tokenizer, device, records, peak_layer):
    # Pooled globally across ALL problems -- unlike this project's usual per-problem-matched-pair
    # convention, a question that consistently only succeeds or only loops (no within-question
    # variance) still contributes its rollouts to the corresponding bucket.
    correct_vecs, looped_vecs = [], []
    for prob in records:
        c, w = prob.get("correct", []), prob.get("looped_wrong", [])
        for rollouts, bucket in [(c, correct_vecs), (w, looped_vecs)]:
            for r in rollouts:
                enc = tokenizer(r["text"], return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
                input_ids = enc["input_ids"].to(device)
                with torch.no_grad():
                    out = model(input_ids, output_hidden_states=True)
                seq_len = input_ids.shape[1]
                start = int(seq_len * 0.75)
                vec = out.hidden_states[peak_layer][0, start:].mean(dim=0).float().cpu().numpy()
                bucket.append(vec)
                del out
    if not correct_vecs or not looped_vecs:
        raise RuntimeError(
            f"Not enough qualifying problems to build a direction: "
            f"{len(correct_vecs)} correct vecs, {len(looped_vecs)} looped vecs collected "
            f"(a problem only counts if it has >=1 correct AND >=1 looped_wrong rollout). "
            f"Increase n_problems or n_rollouts_per_problem."
        )
    correct_vecs, looped_vecs = np.array(correct_vecs), np.array(looped_vecs)
    direction = looped_vecs.mean(axis=0) - correct_vecs.mean(axis=0)
    typical_norm = float(np.linalg.norm(np.concatenate([correct_vecs, looped_vecs]), axis=1).mean())
    unit_direction = (direction / (np.linalg.norm(direction) + 1e-8)).astype(np.float32)
    return unit_direction, typical_norm, len(correct_vecs), len(looped_vecs)


def main():
    model_path = sys.argv[1]
    peak_layer = int(sys.argv[2])
    out_dir = sys.argv[3]
    n_problems = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    n_rollouts = int(sys.argv[5]) if len(sys.argv) > 5 else 8

    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    device = next(model.parameters()).device

    print(f"Sampling {n_rollouts} rollouts/problem for {n_problems} problems "
          f"(temperature={TEMPERATURE}, max_new_tokens={MAX_NEW_TOKENS}) ...", flush=True)
    records = generate_rollouts(model, tokenizer, device, n_problems, n_rollouts)

    rollouts_path = f"{out_dir}/all_rollouts.json"
    with open(rollouts_path, "w") as f:
        json.dump(records, f)
    n_c = sum(len(p["correct"]) for p in records)
    n_l = sum(len(p["looped_wrong"]) for p in records)
    n_t = sum(len(p["terminated_wrong"]) for p in records)
    print(f"Saved {len(records)} problem records to {rollouts_path} "
          f"(correct={n_c}, looped_wrong={n_l}, terminated_wrong={n_t})", flush=True)

    print(f"Building direction at layer {peak_layer} ...", flush=True)
    unit_direction, typical_norm, n_correct_used, n_looped_used = build_direction(
        model, tokenizer, device, records, peak_layer
    )
    np.save(f"{out_dir}/direction.npy", unit_direction)
    meta = {
        "model_path": model_path, "peak_layer": peak_layer, "typical_norm": typical_norm,
        "hidden_size": int(unit_direction.shape[0]), "source_rollouts_path": rollouts_path,
        "direction_semantics": "looped_wrong (cannot stop) minus correct (terminates + right answer)",
        "n_correct_rollouts_used": n_correct_used, "n_looped_wrong_rollouts_used": n_looped_used,
        "n_problems": n_problems, "n_rollouts_per_problem": n_rollouts, "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved direction.npy + meta.json to {out_dir} | typical_norm={typical_norm:.3f} "
          f"n_correct_used={n_correct_used} n_looped_used={n_looped_used}", flush=True)


if __name__ == "__main__":
    main()
