"""
What's actually happening inside completions that never terminate (hit the
max_new_tokens cap)? For every such completion, checks:

  1. Does the correct answer appear ANYWHERE in the text before the cutoff,
     or does the model never produce it at all? Tested by cumulative-decile
     extraction: for each decile of the text (10%, 20%, ..., 100%), run the
     same extraction+comparison used in sparsity_sweep.py on the text-so-far,
     and record the EARLIEST decile at which the correct answer becomes
     extractable. If no decile ever matches, the model never produced it
     ("pure rambling").
  2. A repetitiveness score for the final 20% of the text (zlib compression
     ratio -- highly looping/repeated text compresses far better than
     genuinely novel reasoning).
  3. Self-correction language density in the final quartile (same regex
     used elsewhere in this project), as a proxy for whether the model is
     still "actively working" vs. just producing filler.

Every completion (truncated or not) is saved with its full text to a JSON
file, so you can go back and read specific examples afterward.

Usage:
    python truncation_content_analysis.py <size> <specs> [n_problems]

    <size>      "1.5B" or "7B"
    <specs>     comma-separated list of "<checkpoint>:<alpha_mult>" pairs.
                <checkpoint> is either "dense" (the unpruned model) or a
                sparsity percentage like "50" (maps to the OBC-Prune
                checkpoint sweep_daoc_<size>_sparse<N> -- edit
                CHECKPOINT_PATHS below to point at a C4 checkpoint instead
                if you want to study C4-induced non-termination).
    [n_problems] number of MATH-500 problems to test per setting (default 40)

Example:
    python truncation_content_analysis.py 1.5B dense:0,dense:2,50:-2,50:0,50:2,60:0

This runs 6 settings: the dense model at alpha=0 and alpha=+2, the 50%
OBC-Prune checkpoint at alpha=-2/0/+2, and the 60% checkpoint at alpha=0 --
loading each checkpoint only once even though it's used at multiple alphas.
"""
import gc
import json
import re
import sys
import zlib

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from lighteval.metrics.utils.extractive_match_utils import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    get_extraction_regexes,
    extract_target_from_pred,
)
from lighteval.metrics.utils.math_comparison import compare_gold_target
from lighteval.utils.language import Language

MAX_SEQ_LEN = 4096
TEST_MAX_NEW_TOKENS = 16384
OUT_DIR = "./truncation_analysis"

# Same per-size settings as sparsity_sweep.py -- edit paths for your setup.
MODEL_CONFIGS = {
    "1.5B": {
        "dense_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "rollouts_path": "./models/dense_1.5B_calib_for_steering/all_rollouts.json",
        "peak_layer": 15,
        "batch_size": 10,  # reduced from 40: 16384-token cap means ~4x the KV cache per sequence
        "checkpoint_paths": {  # sparsity_pct (str) -> local path or HF id
            str(sp): f"./models/sweep_daoc_1.5B_sparse{sp}" for sp in [10, 20, 30, 40, 50, 60, 70, 80, 90]
        },
    },
    "7B": {
        "dense_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "rollouts_path": "./models/dense_7B_calib_for_steering/all_rollouts.json",
        "peak_layer": 15,
        "batch_size": 5,  # reduced from 20: 16384-token cap means ~4x the KV cache per sequence
        "checkpoint_paths": {
            str(sp): f"./models/sweep_daoc_7B_sparse{sp}" for sp in [10, 20, 30, 40, 50, 60, 70, 80, 90]
        },
    },
}

MATH_QUERY_TEMPLATE = """
Solve the following problem. The final line of your response MUST be of the following format:
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the final answer. Think step by step before answering.

{prompt}
""".strip()

PRED_REGEXES = get_extraction_regexes(None, [ExprExtractionConfig(), LatexExtractionConfig()], Language.ENGLISH)
GOLD_REGEXES = get_extraction_regexes(None, [ExprExtractionConfig(), LatexExtractionConfig()], Language.ENGLISH)

SELF_CORRECT_RE = re.compile(
    r"\b(wait|actually|hmm|let me re-?check|let me reconsider|i made a mistake|"
    r"that'?s wrong|that is wrong|double.?check|re-?examin|on second thought|"
    r"let me verify|i think i (made|had) an error|correcting|recompute)\b",
    re.IGNORECASE,
)


def is_correct(pred_text, gold_text):
    ep = extract_target_from_pred(pred_text, PRED_REGEXES, "first_match", "any_match", 5)
    eg = extract_target_from_pred(gold_text, GOLD_REGEXES, "first_match", "any_match", 5)
    if not ep or not eg:
        return False
    try:
        return bool(compare_gold_target(eg, ep, precision=6))
    except Exception:
        return False


def earliest_decile_with_correct_answer(text, gold_text):
    n = len(text)
    if n == 0:
        return None
    for d in range(1, 11):
        if is_correct(text[: int(n * d / 10)], gold_text):
            return d
    return None


def repetitiveness_score(text):
    tail = text[-max(1, int(len(text) * 0.2)):].encode("utf-8", errors="ignore")
    if len(tail) < 50:
        return None
    return len(zlib.compress(tail, level=9)) / len(tail)  # lower = more repetitive/compressible


def self_correct_density_last_quartile(text):
    n = len(text)
    if n == 0:
        return 0.0
    tail = text[int(n * 0.75):]
    n_words = max(len(tail.split()), 1)
    return 1000 * len(SELF_CORRECT_RE.findall(tail)) / n_words


def make_hook(direction_tensor, alpha_holder):
    def hook(module, inputs, output):
        if alpha_holder["alpha"] == 0:
            return output
        if isinstance(output, tuple):
            return (output[0] + alpha_holder["alpha"] * direction_tensor,) + output[1:]
        return output + alpha_holder["alpha"] * direction_tensor
    return hook


def compute_direction(dense_model_name, rollouts_path, peak_layer, device):
    print(f"Loading DENSE model {dense_model_name} to build steering direction ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(dense_model_name)
    model = AutoModelForCausalLM.from_pretrained(dense_model_name, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    with open(rollouts_path) as f:
        data = json.load(f)
    correct_vecs, wrong_vecs = [], []
    for prob in data:
        c, w = prob.get("correct", []), prob.get("wrong", [])
        if not c or not w:
            continue
        for cls, rollouts, bucket in [("correct", c[:6], correct_vecs), ("wrong", w[:6], wrong_vecs)]:
            for r in rollouts:
                text = r.get("text", "") if isinstance(r, dict) else str(r)
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
                input_ids = enc["input_ids"].to(device)
                with torch.no_grad():
                    out = model(input_ids, output_hidden_states=True)
                seq_len = input_ids.shape[1]
                start = int(seq_len * 0.75)
                vec = out.hidden_states[peak_layer][0, start:].mean(dim=0).float().cpu().numpy()
                (correct_vecs if cls == "correct" else wrong_vecs).append(vec)
                del out
    correct_vecs, wrong_vecs = np.array(correct_vecs), np.array(wrong_vecs)
    direction = wrong_vecs.mean(axis=0) - correct_vecs.mean(axis=0)
    typical_norm = np.linalg.norm(np.concatenate([correct_vecs, wrong_vecs]), axis=1).mean()
    unit_direction = direction / (np.linalg.norm(direction) + 1e-8)
    print(f"Direction built. typical_norm={typical_norm:.3f}", flush=True)
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return unit_direction, typical_norm


def main():
    size = sys.argv[1]
    specs = sys.argv[2]
    n_problems = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    cfg = MODEL_CONFIGS[size]
    device = torch.device("cuda")

    settings = []
    for spec in specs.split(","):
        ckpt_key, alpha_str = spec.split(":")
        alpha = float(alpha_str)
        alpha = int(alpha) if alpha == int(alpha) else alpha
        ckpt_path = cfg["dense_model"] if ckpt_key == "dense" else cfg["checkpoint_paths"][ckpt_key]
        tag = f"{ckpt_key}_alpha{alpha}"
        settings.append({"tag": tag, "ckpt": ckpt_path, "alpha_mult": alpha})

    direction_np, typical_norm = compute_direction(cfg["dense_model"], cfg["rollouts_path"], cfg["peak_layer"], device)

    ds_test = load_dataset("HuggingFaceH4/MATH-500", split="test")
    test_problems = list(ds_test)[:n_problems]
    print(f"Using {len(test_problems)} MATH-500 problems per setting.", flush=True)

    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    ckpt_cache = {}
    for setting in settings:
        tag, ckpt_path, alpha_mult = setting["tag"], setting["ckpt"], setting["alpha_mult"]
        print(f"\n\n===== SETTING {tag} (ckpt={ckpt_path}, alpha_mult={alpha_mult}) =====", flush=True)

        if ckpt_path not in ckpt_cache:
            for m, t in ckpt_cache.values():
                del m, t
            ckpt_cache.clear()
            torch.cuda.empty_cache()
            gc.collect()
            tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(ckpt_path, dtype=torch.bfloat16, device_map="cuda")
            model.eval()
            ckpt_cache[ckpt_path] = (model, tokenizer)
        model, tokenizer = ckpt_cache[ckpt_path]

        direction_tensor = torch.tensor(direction_np, dtype=torch.bfloat16, device=device)
        alpha_holder = {"alpha": alpha_mult * typical_norm * 0.1}
        target_layer = model.model.layers[cfg["peak_layer"] - 1]
        handle = target_layer.register_forward_hook(make_hook(direction_tensor, alpha_holder))

        input_texts = []
        for prob in test_problems:
            query = MATH_QUERY_TEMPLATE.format(prompt=prob["problem"])
            messages = [{"role": "user", "content": query}]
            input_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        batch_size = cfg["batch_size"]
        records = []
        for batch_start in range(0, len(test_problems), batch_size):
            batch_texts = input_texts[batch_start:batch_start + batch_size]
            batch_problems = test_problems[batch_start:batch_start + batch_size]
            enc = tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN, padding=True)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            prompt_len = input_ids.shape[1]
            with torch.no_grad():
                gen = model.generate(
                    input_ids, attention_mask=attention_mask, max_new_tokens=TEST_MAX_NEW_TOKENS,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id,
                )
            for i, prob in enumerate(batch_problems):
                new_tokens = gen[i, prompt_len:]
                n_nonpad = (new_tokens != tokenizer.pad_token_id).sum().item()
                truncated = n_nonpad >= TEST_MAX_NEW_TOKENS
                completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
                gold = f"ANSWER: {prob['solution']}"
                correct = is_correct(completion, gold)
                record = {
                    "problem": prob["problem"], "gold_solution": prob["solution"], "completion": completion,
                    "truncated": bool(truncated), "final_correct": bool(correct), "n_tokens": int(n_nonpad),
                }
                if truncated:
                    record["earliest_decile_with_correct_answer"] = earliest_decile_with_correct_answer(completion, gold)
                    record["repetitiveness_score_last20pct"] = repetitiveness_score(completion)
                    record["self_correct_density_last_quartile"] = self_correct_density_last_quartile(completion)
                records.append(record)
            del gen, input_ids, attention_mask
            torch.cuda.empty_cache()

        out_path = f"{OUT_DIR}/{tag}.json"
        with open(out_path, "w") as f:
            json.dump(records, f)
        n_truncated = sum(r["truncated"] for r in records)
        n_contains = sum(1 for r in records if r["truncated"] and r.get("earliest_decile_with_correct_answer"))
        print(f"[{tag}] n_truncated={n_truncated}/{len(records)} | "
              f"of those, contains-correct-answer-somewhere={n_contains}/{n_truncated if n_truncated else 1}", flush=True)
        print(f"[{tag}] Saved full completions to {out_path}", flush=True)
        handle.remove()

    print("\n\n===== ALL SETTINGS DONE =====", flush=True)


if __name__ == "__main__":
    main()
