#!/usr/bin/env python3
"""
Steer DeepSeek-R1-Distill-Qwen-1.5B (OBC-Prune, 60% sparsity) with the
loop-vs-correct direction built by sample_pruned_loop_direction.py, and
evaluate on held-out MATH-500 problems.

Sign convention: direction = mean(looped_wrong) - mean(correct), so it
points TOWARD "loops forever". Steering with a NEGATIVE alpha_mult pushes
AWAY from looping (hypothesis: lower truncation rate, maybe higher
accuracy); POSITIVE alpha_mult pushes TOWARD looping (hypothesis: higher
truncation rate).

Usage:
    python steer_with_loop_direction.py <model_path> <direction_dir> <alphas> [n_problems]
"""
import json
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
BATCH_SIZE = 8

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


def main():
    model_path = sys.argv[1]
    direction_dir = sys.argv[2]
    alphas = [float(a) for a in sys.argv[3].split(",")]
    n_problems = int(sys.argv[4]) if len(sys.argv) > 4 else 100

    unit_direction = np.load(f"{direction_dir}/direction.npy")
    meta = json.load(open(f"{direction_dir}/meta.json"))
    peak_layer, typical_norm = meta["peak_layer"], meta["typical_norm"]
    print(f"Loaded direction: peak_layer={peak_layer} typical_norm={typical_norm:.3f} "
          f"(n_correct_used={meta['n_correct_rollouts_used']}, n_looped_used={meta['n_looped_wrong_rollouts_used']})",
          flush=True)

    print(f"Loading {model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    device = next(model.parameters()).device

    direction_tensor = torch.tensor(unit_direction, dtype=torch.bfloat16, device=device)
    target_layer = model.model.layers[peak_layer - 1]

    ds_test = load_dataset("HuggingFaceH4/MATH-500", split="test")
    test_problems = list(ds_test)[:n_problems]
    print(f"Using {len(test_problems)} MATH-500 problems.", flush=True)

    input_texts = []
    for prob in test_problems:
        query = MATH_QUERY_TEMPLATE.format(prompt=prob["problem"])
        messages = [{"role": "user", "content": query}]
        input_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    results = []
    for alpha_mult in alphas:
        alpha = alpha_mult * typical_norm * 0.1

        def hook(module, inputs, output, _alpha=alpha):
            if _alpha == 0:
                return output
            if isinstance(output, tuple):
                return (output[0] + _alpha * direction_tensor,) + output[1:]
            return output + _alpha * direction_tensor

        handle = target_layer.register_forward_hook(hook)

        n_correct, n_truncated, n_rep_stopped, n = 0, 0, 0, 0
        for batch_start in range(0, len(test_problems), BATCH_SIZE):
            batch_texts = input_texts[batch_start:batch_start + BATCH_SIZE]
            batch_problems = test_problems[batch_start:batch_start + BATCH_SIZE]
            enc = tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN, padding=True)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            prompt_len = input_ids.shape[1]
            rep_criteria = RepetitionStoppingCriteria(tokenizer, prompt_len)
            with torch.no_grad():
                gen = model.generate(
                    input_ids, attention_mask=attention_mask, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=StoppingCriteriaList([rep_criteria]),
                )
            for i, prob in enumerate(batch_problems):
                new_tokens = gen[i, prompt_len:]
                n_nonpad = (new_tokens != tokenizer.pad_token_id).sum().item()
                repetition_stopped = bool(rep_criteria.stopped and rep_criteria.stopped[i])
                truncated = n_nonpad >= MAX_NEW_TOKENS or repetition_stopped
                completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
                gold = f"ANSWER: {prob['solution']}"
                if is_correct(completion, gold):
                    n_correct += 1
                if truncated:
                    n_truncated += 1
                if repetition_stopped:
                    n_rep_stopped += 1
                n += 1
            del gen, input_ids, attention_mask
            torch.cuda.empty_cache()
            print(f"  alpha_mult={alpha_mult} {n}/{len(test_problems)} done "
                  f"(acc so far={n_correct/n:.3f}, trunc so far={n_truncated/n:.3f})", flush=True)

        handle.remove()
        acc = n_correct / n
        trunc_rate = n_truncated / n
        print(f"alpha_mult={alpha_mult:+.1f} (raw alpha={alpha:.2f}): acc={acc:.3f} trunc_rate={trunc_rate:.3f} "
              f"(rep_stopped={n_rep_stopped}/{n}) (n={n})", flush=True)
        results.append({"alpha_mult": alpha_mult, "raw_alpha": alpha, "accuracy": acc,
                         "truncation_rate": trunc_rate, "n_rep_stopped": n_rep_stopped, "n": n})

    with open(f"{direction_dir}/steering_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {direction_dir}/steering_eval_results.json", flush=True)
    print("\nSummary:")
    for r in results:
        print(f"  alpha_mult={r['alpha_mult']:+.1f}: acc={r['accuracy']:.3f} trunc={r['truncation_rate']:.3f}")


if __name__ == "__main__":
    main()
