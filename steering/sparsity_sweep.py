"""
Scaled-up magnitude-matched steering check: n=100 MATH-500 problems
(instead of 30), greedy decoding, across all 38 checkpoints already
established (dense + 9 OBC-Prune + 9 C4 sparsities, per size). Alpha
values to run are passed via argv, in phases:
  Phase 1: -2, 0, 2   (re-run at n=100; supersedes the n=30 fig14 data)
  Phase 2: -1, 1
  Phase 3: -4, 4

Reuses the saved calibration rollouts and already-established peak layer
15 for both sizes -- no calibration regeneration or layer sweep needed.

Usage: sparsity_sweep_n100.py <size: 1.5B|7B> <comma-separated alphas>
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
TEST_MAX_NEW_TOKENS = 16384

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
        "batch_size_test": 12,  # reduced from 50: 16384-token cap means ~4x the KV cache per sequence
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
        "batch_size_test": 6,  # reduced from 25: 16384-token cap means ~4x the KV cache per sequence
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


def compute_direction_from_saved_rollouts(model, tokenizer, device, rollouts_path, peak_layer, size_label):
    print(f"[{size_label}] Building direction at layer {peak_layer} from {rollouts_path} ...", flush=True)
    with open(rollouts_path) as f:
        data = json.load(f)
    correct_vecs, wrong_vecs = [], []
    for prob in data:
        c = prob.get("correct", [])
        w = prob.get("wrong", [])
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
                bucket.append(vec)
                del out
    correct_vecs = np.array(correct_vecs)
    wrong_vecs = np.array(wrong_vecs)
    direction = wrong_vecs.mean(axis=0) - correct_vecs.mean(axis=0)
    typical_norm = np.linalg.norm(np.concatenate([correct_vecs, wrong_vecs]), axis=1).mean()
    unit_direction = direction / (np.linalg.norm(direction) + 1e-8)
    print(f"[{size_label}] Direction built from {len(correct_vecs)} correct + {len(wrong_vecs)} wrong rollouts. "
          f"typical_norm={typical_norm:.3f}", flush=True)
    torch.cuda.empty_cache()
    gc.collect()
    return unit_direction, typical_norm


def run_sweep_on_checkpoint(model, tokenizer, direction_np, typical_norm, peak_layer, test_problems,
                             batch_size, model_label, alphas):
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

    for alpha_mult in alphas:
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
    alphas = [float(a) for a in sys.argv[2].split(",")]
    alphas = [int(a) if a == int(a) else a for a in alphas]
    only_tags = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
    cfg = MODEL_CONFIGS[size_label]
    device = torch.device("cuda")

    print(f"\n\n########################## SIZE {size_label} (n={N_TEST_PROBLEMS}, alphas={alphas}) ##########################",
          flush=True)
    print(f"Loading DENSE {cfg['dense_model']} (to build direction from saved rollouts) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["dense_model"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["dense_model"], dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    direction_np, typical_norm = compute_direction_from_saved_rollouts(
        model, tokenizer, device, cfg["rollouts_path"], cfg["peak_layer"], size_label)

    ds_test = load_dataset("HuggingFaceH4/MATH-500", split="test")
    test_problems = list(ds_test)[:N_TEST_PROBLEMS]
    print(f"[{size_label}] Using {len(test_problems)} MATH-500 problems.", flush=True)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    checkpoints = cfg["checkpoints"] if only_tags is None else [(t, p) for t, p in cfg["checkpoints"] if t in only_tags]
    for tag, ckpt_path in checkpoints:
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
                                 test_problems, cfg["batch_size_test"], f"{size_label}-{tag}", alphas)
        del p_model, p_tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n===== [{size_label}] PHASE DONE (alphas={alphas}) =====", flush=True)


if __name__ == "__main__":
    main()
