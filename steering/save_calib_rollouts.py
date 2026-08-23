"""
Regenerates the calibration rollouts used to build the steering direction
for DeepSeek-R1-Distill-Qwen-1.5B and -7B in §4.7 (sparsity_sweep_1p5b_7b.py),
this time SAVING the output to disk in the same all_rollouts.json schema
used elsewhere in the project (list of {problem_id, problem, difficulty,
correct: [{text, token_count}], wrong: [{text, token_count}]}).

Same generation settings as the original run, for exact reproducibility:
40 OpenR1-Math-220k problems with extractable gold answers, 6 rollouts per
problem (temperature=0.8), max_new_tokens=2560.
"""
import json

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
N_CALIB_PROBLEMS = 40
N_ROLLOUTS_PER_PROBLEM = 6
CALIB_TEMPERATURE = 0.8
CALIB_MAX_NEW_TOKENS = 2560

MATH_QUERY_TEMPLATE = """
Solve the following problem. The final line of your response MUST be of the following format:
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the final answer. Think step by step before answering.

{prompt}
""".strip()

PRED_REGEXES = get_extraction_regexes(None, [ExprExtractionConfig(), LatexExtractionConfig()], Language.ENGLISH)
GOLD_REGEXES = get_extraction_regexes(None, [ExprExtractionConfig(), LatexExtractionConfig()], Language.ENGLISH)

MODEL_CONFIGS = [
    {
        "size": "1.5B",
        "dense_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "out_path": "/mnt/data/lannth/COMP/RAC/RAC/open-r1-main/models/dense_1.5B_calib_for_steering/all_rollouts.json",
    },
    {
        "size": "7B",
        "dense_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "out_path": "/mnt/data/lannth/COMP/RAC/RAC/open-r1-main/models/dense_7B_calib_for_steering/all_rollouts.json",
    },
]


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


def generate_and_save(cfg):
    size_label = cfg["size"]
    print(f"\n\n########## SIZE {size_label} ##########", flush=True)
    print(f"Loading DENSE {cfg['dense_model']} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["dense_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["dense_model"], dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    device = next(model.parameters()).device

    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train")
    candidates = []
    candidate_indices = []
    for i in range(len(ds)):
        if len(candidates) >= N_CALIB_PROBLEMS:
            break
        row = ds[i]
        gold = f"ANSWER: {row['answer']}"
        if gold_extractable(gold):
            candidates.append(row)
            candidate_indices.append(i)
    print(f"[{size_label}] Selected {len(candidates)} problems with extractable gold answers", flush=True)

    out_records = []
    for pi, (row, ds_idx) in enumerate(zip(candidates, candidate_indices)):
        query = MATH_QUERY_TEMPLATE.format(prompt=row["problem"])
        messages = [{"role": "user", "content": query}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            gen = model.generate(
                input_ids,
                max_new_tokens=CALIB_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=CALIB_TEMPERATURE,
                num_return_sequences=N_ROLLOUTS_PER_PROBLEM,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        prompt_len = input_ids.shape[1]
        gold = f"ANSWER: {row['answer']}"
        correct_rollouts, wrong_rollouts = [], []
        for r in range(gen.shape[0]):
            new_tokens = gen[r, prompt_len:]
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
            full_text = prompt_text + completion
            token_count = int((new_tokens != (tokenizer.pad_token_id or tokenizer.eos_token_id)).sum().item())
            rollout_record = {"text": full_text, "token_count": token_count}
            if is_correct(completion, gold):
                correct_rollouts.append(rollout_record)
            else:
                wrong_rollouts.append(rollout_record)
        n_total = len(correct_rollouts) + len(wrong_rollouts)
        difficulty = (len(wrong_rollouts) / n_total) if n_total else None
        out_records.append({
            "problem_id": ds_idx,
            "problem": row["problem"],
            "difficulty": difficulty,
            "correct": correct_rollouts,
            "wrong": wrong_rollouts,
        })
        if (pi + 1) % 10 == 0:
            n_c = sum(len(p["correct"]) for p in out_records)
            n_w = sum(len(p["wrong"]) for p in out_records)
            print(f"  [{size_label}] {pi+1}/{len(candidates)} problems done ({n_c} correct, {n_w} wrong so far)",
                  flush=True)

    qualifying = [p for p in out_records if p["correct"] and p["wrong"]]
    n_c = sum(len(p["correct"]) for p in qualifying)
    n_w = sum(len(p["wrong"]) for p in qualifying)
    print(f"[{size_label}] Done: {len(qualifying)}/{len(out_records)} problems qualify (>=1 correct & >=1 wrong), "
          f"{n_c} correct + {n_w} wrong rollouts total", flush=True)

    import os
    os.makedirs(os.path.dirname(cfg["out_path"]), exist_ok=True)
    with open(cfg["out_path"], "w") as f:
        json.dump(out_records, f)
    print(f"[{size_label}] Saved {len(out_records)} problem records to {cfg['out_path']}", flush=True)

    del model
    torch.cuda.empty_cache()


def main():
    for cfg in MODEL_CONFIGS:
        generate_and_save(cfg)
    print("\n\n===== ALL SIZES SAVED =====", flush=True)


if __name__ == "__main__":
    main()
