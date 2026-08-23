"""
Length-matched-direction ablation: the steering direction used throughout
this investigation (mean(wrong last-quarter-mean hidden state) -
mean(correct last-quarter-mean hidden state)) was built from ALL
available correct/wrong rollouts with no length-based selection -- but
the underlying data has a real length gap (wrong rollouts run longer on
average, an established finding from the same-question analysis). This
script tests whether that gap confounds the direction, by:

  1. Pooling all correct and wrong rollouts from the saved calibration
     data (same source as sparsity_sweep_n100.py), each with its
     token_count.
  2. Greedily pairing each wrong rollout with the closest-length unused
     correct rollout (by token_count), discarding any leftover
     unmatched rollouts from whichever pool is larger.
  3. Building the direction from ONLY the matched pairs (same peak layer
     15, same last-quarter-mean methodology) -- so correct and wrong
     summary vectors now come from length-controlled populations.
  4. Re-running the EXACT same Phase 1 sweep as
     new_paper_reasoning_signatures/figures/fig15_n100_phase1_both.png:
     alpha_mult in {-2, 0, +2}, n=100 MATH-500 problems, same 19
     checkpoints per size (dense + 9 OBC-Prune + 9 C4 sparsities), same
     batch sizes -- so results are directly comparable to the existing
     unmatched-direction figure.

Usage: sparsity_sweep_length_matched.py <size: 1.5B|7B>
"""
import gc
import json
import sys

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
N_TEST_PROBLEMS = 100
TEST_MAX_NEW_TOKENS = 4096
ALPHAS = [-2, 0, 2]
MAX_ROLLOUTS_PER_CLASS_PER_PROBLEM = 6

MATH_QUERY_TEMPLATE = """
Solve the following problem. The final line of your response MUST be of the following format:
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the final answer. Think step by step before answering.

{prompt}
""".strip()

PRED_REGEXES = get_extraction_regexes(None, [ExprExtractionConfig(), LatexExtractionConfig()], Language.ENGLISH)
GOLD_REGEXES = get_extraction_regexes(None, [ExprExtractionConfig(), LatexExtractionConfig()], Language.ENGLISH)

SPARSITIES = [10, 20, 30, 40, 50, 60, 70, 80, 90]

MODEL_CONFIGS = {
    "1.5B": {
        "dense_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "rollouts_path": "/mnt/data/lannth/COMP/RAC/RAC/open-r1-main/models/dense_1.5B_calib_for_steering/all_rollouts.json",
        "peak_layer": 15,
        "batch_size_test": 50,
        "checkpoints": (
            [("dense", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")]
            + [(f"obc_{sp}", f"/mnt/data/lannth/COMP/RAC/RAC/open-r1-main/models/sweep_daoc_1.5B_sparse{sp}") for sp in SPARSITIES]
            + [(f"c4_{sp}", f"/mnt/data/lannth/COMP/RAC/RAC/open-r1-main/models/sweep_c4_1.5B_sparse{sp}") for sp in SPARSITIES]
        ),
    },
    "7B": {
        "dense_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "rollouts_path": "/mnt/data/lannth/COMP/RAC/RAC/open-r1-main/models/dense_7B_calib_for_steering/all_rollouts.json",
        "peak_layer": 15,
        "batch_size_test": 25,
        "checkpoints": (
            [("dense", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")]
            + [(f"obc_{sp}", f"/mnt/data/lannth/COMP/RAC/RAC/open-r1-main/models/sweep_daoc_7B_sparse{sp}") for sp in SPARSITIES]
            + [(f"c4_{sp}", f"/mnt/data/lannth/COMP/RAC/RAC/open-r1-main/models/sweep_c4_7B_sparse{sp}") for sp in SPARSITIES]
        ),
    },
}


def is_correct(pred_text, gold_text):
    ep = extract_target_from_pred(pred_text, PRED_REGEXES, "first_match", "any_match", 5)
    eg = extract_target_from_pred(gold_text, GOLD_REGEXES, "first_match", "any_match", 5)
    if not ep or not eg:
        return False
    try:
        return bool(compare_gold_target(eg, ep, precision=6))
    except Exception:
        return False


def make_hook(direction_tensor, alpha_holder):
    def hook(module, inputs, output):
        if alpha_holder["alpha"] == 0:
            return output
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + alpha_holder["alpha"] * direction_tensor
            return (hidden,) + output[1:]
        else:
            return output + alpha_holder["alpha"] * direction_tensor
    return hook


def greedy_length_match(correct_pool, wrong_pool):
    """correct_pool/wrong_pool: list of (text, token_count). Returns
    (matched_correct, matched_wrong) lists of texts, paired by closest
    token_count, greedily, largest-first (to avoid the largest items
    being left with only bad matches at the end)."""
    correct_remaining = sorted(correct_pool, key=lambda x: x[1])
    wrong_remaining = sorted(wrong_pool, key=lambda x: x[1])
    # process the shorter list fully, matching each to nearest in the other
    if len(wrong_remaining) <= len(correct_remaining):
        driver, other = wrong_remaining, correct_remaining
        driver_is_wrong = True
    else:
        driver, other = correct_remaining, wrong_remaining
        driver_is_wrong = False

    other_tc = [x[1] for x in other]
    used = [False] * len(other)
    matched_pairs = []
    diffs = []
    for text, tc in driver:
        best_idx, best_diff = None, None
        for i, otc in enumerate(other_tc):
            if used[i]:
                continue
            d = abs(otc - tc)
            if best_diff is None or d < best_diff:
                best_diff = d
                best_idx = i
        used[best_idx] = True
        diffs.append(best_diff)
        if driver_is_wrong:
            matched_pairs.append((other[best_idx][0], text))  # (correct_text, wrong_text)
        else:
            matched_pairs.append((text, other[best_idx][0]))

    matched_correct = [c for c, w in matched_pairs]
    matched_wrong = [w for c, w in matched_pairs]
    return matched_correct, matched_wrong, diffs


def build_length_matched_direction(model, tokenizer, device, rollouts_path, peak_layer, size_label):
    print(f"[{size_label}] Loading rollouts from {rollouts_path} for length-matched pairing ...", flush=True)
    with open(rollouts_path) as f:
        data = json.load(f)
    correct_pool, wrong_pool = [], []
    for prob in data:
        c = prob.get("correct", [])
        w = prob.get("wrong", [])
        if not c or not w:
            continue
        for r in c[:MAX_ROLLOUTS_PER_CLASS_PER_PROBLEM]:
            correct_pool.append((r.get("text", ""), r.get("token_count", len(r.get("text", "")))))
        for r in w[:MAX_ROLLOUTS_PER_CLASS_PER_PROBLEM]:
            wrong_pool.append((r.get("text", ""), r.get("token_count", len(r.get("text", "")))))

    orig_mean_correct = np.mean([tc for _, tc in correct_pool])
    orig_mean_wrong = np.mean([tc for _, tc in wrong_pool])
    print(f"[{size_label}] BEFORE matching: n_correct={len(correct_pool)} mean_tc={orig_mean_correct:.1f} | "
          f"n_wrong={len(wrong_pool)} mean_tc={orig_mean_wrong:.1f} | gap={orig_mean_wrong-orig_mean_correct:+.1f}",
          flush=True)

    matched_correct_texts, matched_wrong_texts, diffs = greedy_length_match(correct_pool, wrong_pool)
    n_matched = len(matched_correct_texts)
    print(f"[{size_label}] AFTER matching: n_pairs={n_matched} | mean_abs_token_diff_within_pair={np.mean(diffs):.1f} "
          f"| median={np.median(diffs):.1f} | max={np.max(diffs):.1f}", flush=True)

    correct_vecs, wrong_vecs = [], []
    for cls, texts, bucket in [("correct", matched_correct_texts, correct_vecs),
                                 ("wrong", matched_wrong_texts, wrong_vecs)]:
        for text in texts:
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
    typical_norm = np.linalg.norm(np.concatenate([correct_vecs, wrong_vecs]), axis=1).mean()
    unit_direction = direction / (np.linalg.norm(direction) + 1e-8)
    print(f"[{size_label}] Length-matched direction built from {len(correct_vecs)} correct + {len(wrong_vecs)} wrong "
          f"MATCHED rollouts. typical_norm={typical_norm:.3f}", flush=True)
    torch.cuda.empty_cache()
    gc.collect()
    return unit_direction, typical_norm


def run_sweep_on_checkpoint(model, tokenizer, direction_np, typical_norm, peak_layer, test_problems,
                             batch_size, model_label):
    device = next(model.parameters()).device
    direction_tensor = torch.tensor(direction_np, dtype=torch.bfloat16, device=device)

    input_texts = []
    for prob in test_problems:
        query = MATH_QUERY_TEMPLATE.format(prompt=prob["problem"])
        messages = [{"role": "user", "content": query}]
        input_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    alpha_holder = {"alpha": 0}
    target_layer = model.model.layers[peak_layer - 1] if peak_layer > 0 else model.model.embed_tokens
    handle = target_layer.register_forward_hook(make_hook(direction_tensor, alpha_holder))

    for alpha_mult in ALPHAS:
        alpha = alpha_mult * typical_norm * 0.1
        alpha_holder["alpha"] = alpha
        n_correct = 0
        n_correct_natural = 0
        n_truncated = 0
        gen_lengths_natural = []
        for batch_start in range(0, len(test_problems), batch_size):
            batch_texts = input_texts[batch_start:batch_start + batch_size]
            batch_problems = test_problems[batch_start:batch_start + batch_size]
            enc = tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN, padding=True)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            prompt_len = input_ids.shape[1]
            with torch.no_grad():
                gen = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=TEST_MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for i, prob in enumerate(batch_problems):
                new_tokens = gen[i, prompt_len:]
                n_nonpad = (new_tokens != tokenizer.pad_token_id).sum().item()
                truncated = n_nonpad >= TEST_MAX_NEW_TOKENS
                completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
                gold = f"ANSWER: {prob['solution']}"
                correct = is_correct(completion, gold)
                n_correct += correct
                if truncated:
                    n_truncated += 1
                else:
                    gen_lengths_natural.append(n_nonpad)
                    n_correct_natural += correct
            del gen, input_ids, attention_mask
            torch.cuda.empty_cache()
        n = len(test_problems)
        n_natural = n - n_truncated
        acc = n_correct / n
        acc_natural = (n_correct_natural / n_natural) if n_natural else float("nan")
        print(f"[{model_label}] alpha_mult={alpha_mult:+.1f}: acc_all={acc:.3f} | acc_natural={acc_natural:.3f} | "
              f"n_truncated={n_truncated}/{n}", flush=True)

    handle.remove()
    print(f"[{model_label}] SWEEP DONE.", flush=True)


def main():
    size_label = sys.argv[1]
    cfg = MODEL_CONFIGS[size_label]
    device = torch.device("cuda")

    print(f"\n\n########################## SIZE {size_label} (LENGTH-MATCHED direction, n={N_TEST_PROBLEMS}, "
          f"alphas={ALPHAS}) ##########################", flush=True)
    print(f"Loading DENSE {cfg['dense_model']} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["dense_model"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["dense_model"], dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    direction_np, typical_norm = build_length_matched_direction(
        model, tokenizer, device, cfg["rollouts_path"], cfg["peak_layer"], size_label)

    ds_test = load_dataset("HuggingFaceH4/MATH-500", split="test")
    test_problems = list(ds_test)[:N_TEST_PROBLEMS]
    print(f"[{size_label}] Using {len(test_problems)} MATH-500 problems.", flush=True)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    for tag, ckpt_path in cfg["checkpoints"]:
        print(f"\n===== [{size_label}] {tag} ({ckpt_path}) =====", flush=True)
        try:
            p_tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
            p_tokenizer.padding_side = "left"
            if p_tokenizer.pad_token is None:
                p_tokenizer.pad_token = p_tokenizer.eos_token
            p_model = AutoModelForCausalLM.from_pretrained(ckpt_path, dtype=torch.bfloat16, device_map="cuda")
            p_model.eval()
        except Exception as e:
            print(f"[{size_label}] FAILED to load {tag}: {e}", flush=True)
            continue
        run_sweep_on_checkpoint(p_model, p_tokenizer, direction_np, typical_norm, cfg["peak_layer"],
                                 test_problems, cfg["batch_size_test"], f"{size_label}-{tag}")
        del p_model, p_tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n===== [{size_label}] LENGTH-MATCHED PHASE DONE =====", flush=True)


if __name__ == "__main__":
    main()
