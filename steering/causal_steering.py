"""
Causal steering experiment: does intervening on the "wrong direction" found
in the hidden-representation analysis actually change generation outcomes,
or is it merely correlational?

Setup:
  - Model: DeepSeek-R1-Distill-Llama-8B, layer 12 (peak layer from the scaled
    hidden-direction analysis).
  - Direction: mean(wrong-rollout last-quarter hidden states) minus
    mean(correct-rollout last-quarter hidden states), built from ALL
    available calibration rollouts (same construction as before, just no
    train/test split now -- we want the best estimate of the direction for
    the intervention, and test on a DIFFERENT problem set: MATH500).
  - Intervention: a forward hook on the target decoder layer adds
    alpha * unit_direction to every position's hidden state, for the full
    generation (prompt processing + every incremental decoding step).
  - Test set: MATH500 problems (held out from calibration), graded with the
    same extraction+comparison pipeline validated earlier this session.
  - Sweep alpha over a range including 0 (unsteered baseline). If accuracy
    increases as alpha becomes more negative (steering away from "wrong")
    and decreases as alpha becomes more positive (steering toward "wrong"),
    that's causal evidence, not just correlation.
"""
import gc
import json
import re

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList

from repetition_stopping import RepetitionStoppingCriteria

from lighteval.metrics.utils.extractive_match_utils import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    get_extraction_regexes,
    extract_target_from_pred,
)
from lighteval.metrics.utils.math_comparison import compare_gold_target
from lighteval.utils.language import Language

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
ROLLOUTS_PATH = "/mnt/data/lannth/COMP/RAC/RAC/open-r1-main/models/llama8B_daoc_sparse04/all_rollouts.json"
LAYER = 12
MAX_SEQ_LEN = 4096
N_TEST_PROBLEMS = 50
MAX_NEW_TOKENS = 16384  # project-wide standard cap (was 4096); tracked explicitly below via truncated flag
# NOTE: this script generates all N_TEST_PROBLEMS in one unbatched call (no chunking, unlike
# sparsity_sweep.py etc.) -- fine at the old 4096 cap on an 80GB GPU, but the ~4x larger KV cache
# at 16384 may need this loop split into sub-batches if you hit OOM.
ALPHAS = [-8, -4, 0, 4, 8]

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


def rollout_text(r):
    return r.get("text", "") if isinstance(r, dict) else str(r)


def compute_direction(model, tokenizer, device):
    print("Computing steering direction from ALL calibration rollouts...", flush=True)
    with open(ROLLOUTS_PATH) as f:
        data = json.load(f)
    correct_vecs, wrong_vecs = [], []
    for prob in data:
        c = prob.get("correct", [])
        w = prob.get("wrong", [])
        if not c or not w:
            continue
        for cls, rollouts, bucket in [("correct", c[:6], correct_vecs), ("wrong", w[:6], wrong_vecs)]:
            for r in rollouts:
                text = rollout_text(r)
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
                input_ids = enc["input_ids"].to(device)
                with torch.no_grad():
                    out = model(input_ids, output_hidden_states=True)
                seq_len = input_ids.shape[1]
                start = int(seq_len * 0.75)
                vec = out.hidden_states[LAYER][0, start:].mean(dim=0).float().cpu().numpy()
                bucket.append(vec)
                del out
    correct_vecs = np.array(correct_vecs)
    wrong_vecs = np.array(wrong_vecs)
    direction = wrong_vecs.mean(axis=0) - correct_vecs.mean(axis=0)
    typical_norm = np.linalg.norm(np.concatenate([correct_vecs, wrong_vecs]), axis=1).mean()
    unit_direction = direction / (np.linalg.norm(direction) + 1e-8)
    print(f"Direction built from {len(correct_vecs)} correct + {len(wrong_vecs)} wrong rollouts. "
          f"Raw diff norm={np.linalg.norm(direction):.3f}, typical hidden-state norm={typical_norm:.3f}", flush=True)
    torch.cuda.empty_cache()
    gc.collect()
    return torch.tensor(unit_direction, dtype=torch.bfloat16, device=device), typical_norm


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


def main():
    print(f"Loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    device = next(model.parameters()).device
    print("Model loaded.", flush=True)

    direction_tensor, typical_norm = compute_direction(model, tokenizer, device)

    print("Loading MATH-500 test problems...", flush=True)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    test_problems = list(ds)[:N_TEST_PROBLEMS]
    print(f"Using {len(test_problems)} MATH500 problems (held out from calibration).", flush=True)

    input_texts = []
    for prob in test_problems:
        query = MATH_QUERY_TEMPLATE.format(prompt=prob["problem"])
        messages = [{"role": "user", "content": query}]
        input_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    enc = tokenizer(input_texts, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN, padding=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    alpha_holder = {"alpha": 0}
    target_layer = model.model.layers[LAYER - 1] if LAYER > 0 else model.model.embed_tokens
    handle = target_layer.register_forward_hook(make_hook(direction_tensor, alpha_holder))

    results = {}
    for alpha_mult in ALPHAS:
        alpha = alpha_mult * typical_norm * 0.1  # scale relative to typical activation norm
        alpha_holder["alpha"] = alpha
        print(f"\n===== alpha_mult={alpha_mult} (raw alpha={alpha:.2f}) =====", flush=True)
        rep_criteria = RepetitionStoppingCriteria(tokenizer, prompt_len)
        with torch.no_grad():
            gen = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([rep_criteria]),
            )
        n_correct = 0
        n_correct_natural = 0
        n_truncated = 0
        gen_lengths = []
        gen_lengths_natural = []
        for i, prob in enumerate(test_problems):
            new_tokens = gen[i, prompt_len:]
            n_nonpad = (new_tokens != tokenizer.pad_token_id).sum().item()
            repetition_stopped = bool(rep_criteria.stopped and rep_criteria.stopped[i])
            # used the full budget, or repetition-loop detector cut it short -> cut off mid-generation either way
            truncated = n_nonpad >= MAX_NEW_TOKENS or repetition_stopped
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
            gen_lengths.append(n_nonpad)
            gold = f"ANSWER: {prob['solution']}"
            correct = is_correct(completion, gold)
            n_correct += correct
            if truncated:
                n_truncated += 1
            else:
                gen_lengths_natural.append(n_nonpad)
                n_correct_natural += correct
        n = len(test_problems)
        n_natural = n - n_truncated
        acc = n_correct / n
        acc_natural = (n_correct_natural / n_natural) if n_natural else float("nan")
        results[alpha_mult] = {
            "acc": acc, "n_correct": n_correct,
            "acc_natural": acc_natural, "n_correct_natural": n_correct_natural, "n_natural": n_natural,
            "n_truncated": n_truncated,
            "mean_gen_len": np.mean(gen_lengths),
            "mean_gen_len_natural": np.mean(gen_lengths_natural) if gen_lengths_natural else float("nan"),
        }
        print(f"alpha_mult={alpha_mult}: accuracy(all)={acc:.3f} ({n_correct}/{n}) | "
              f"accuracy(naturally-finished only)={acc_natural:.3f} ({n_correct_natural}/{n_natural}) | "
              f"truncated={n_truncated}/{n} | mean_gen_tokens(all)={np.mean(gen_lengths):.1f}", flush=True)

    handle.remove()

    print("\n\n===== FINAL SUMMARY =====", flush=True)
    print("alpha_mult  raw_alpha  acc_all  acc_natural_only  n_truncated  mean_gen_tokens(natural)")
    for alpha_mult in ALPHAS:
        r = results[alpha_mult]
        raw_alpha = alpha_mult * typical_norm * 0.1
        print(f"  {alpha_mult:+4d}      {raw_alpha:8.2f}   {r['acc']:.3f}      {r['acc_natural']:.3f}"
              f"            {r['n_truncated']:2d}          {r['mean_gen_len_natural']:.1f}")


if __name__ == "__main__":
    main()
