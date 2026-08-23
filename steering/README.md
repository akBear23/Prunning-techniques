# Activation Steering & Truncation Analysis

Code for the **activation-steering** experiments run on top of the pruned checkpoints in
[`../`](..) — this is the causal follow-up to the pruning work: given a "correct vs. wrong
reasoning" direction found in a model's hidden states, does *forcibly injecting* that direction
during generation actually change accuracy, and does that effect interact with how heavily (and
how) the model was pruned?

If you just want the one-paragraph mental model: **build a direction, add it to a layer's hidden
state during generation, see if accuracy moves.** Everything below is variations on that.

## The method, in one picture

1. Take a pool of *correct* and *wrong* reasoning rollouts from the model (its own generations on
   calibration problems, graded against ground truth).
2. For each rollout, run it through the model and take the **mean hidden state over the last
   quarter of the sequence** at one specific decoder layer ("peak layer", found empirically —
   layer 15 for the 1.5B/7B Qwen-distill models, layer 12 for the Llama-8B pilot).
3. `direction = mean(wrong vectors) − mean(correct vectors)`, normalized to a unit vector.
4. At generation time, register a forward hook on that decoder layer:
   `hidden_state ← hidden_state + alpha * unit_direction`, where
   `alpha = alpha_mult * typical_activation_norm * 0.1`. `alpha_mult` is the human-facing knob —
   negative pushes generation *away* from the "wrong" direction, positive pushes *toward* it,
   `alpha_mult=0` is the unsteered baseline.
5. Generate on a **held-out** problem set (MATH-500) and measure whether accuracy moves with
   `alpha_mult`, across a range of pruning sparsities and both pruning methods (OBC-Prune and C4).

## Prerequisites

- A pruned checkpoint sweep already produced by the code in [`../`](..) (or just the dense/HF
  model — several scripts here include the dense model as `alpha_mult`-swept baseline).
- Saved calibration rollouts in `all_rollouts.json` format:
  `[{"problem_id", "problem", "difficulty", "correct": [{"text", "token_count"}, ...], "wrong": [...]}, ...]`.
  If you don't have these yet, generate them first with `save_calib_rollouts.py` — every other
  script in this folder loads a rollouts file rather than regenerating it, since regenerating is
  the slow part (it means running the *unpruned* model on ~40 calibration problems x 6 rollouts
  each).
- `torch`, `transformers`, `datasets`, and `lighteval` (used only for its answer-extraction /
  math-comparison utilities — `lighteval.metrics.utils.extractive_match_utils` and
  `.math_comparison` — to grade completions against MATH-500 gold answers).

**Heads up on paths:** these scripts were run on a specific H100 cluster and still contain some
hardcoded absolute paths (`/mnt/data/lannth/...`) and conda env references from that environment.
Search for `MODEL_CONFIGS` / `ROLLOUTS_PATH` / `_PATH` constants near the top of each file and
point them at your own checkpoint/rollout locations before running.

## Files, in the order you'd actually use them

| Script | What it does |
|---|---|
| [`save_calib_rollouts.py`](save_calib_rollouts.py) | **Step 0.** Generates calibration rollouts (6 per problem, 40 problems from OpenR1-Math-220k, temperature 0.8) for the dense 1.5B/7B models and saves them to `all_rollouts.json`. Run this once per model size; every script below reuses its output. |
| [`causal_steering.py`](causal_steering.py) | The original, single-model pilot: DeepSeek-R1-Distill-**Llama-8B**, layer 12, alphas `[-8, -4, 0, 4, 8]`, n=50 MATH-500 problems. The simplest, most self-contained example to read first if you want to understand the method end-to-end before looking at the more parametrized sweep scripts below. |
| [`sparsity_sweep.py`](sparsity_sweep.py) | **The main workhorse.** Sweeps a chosen alpha list across dense + all 9 OBC-Prune sparsities (10-90%) + all 9 C4 sparsities (10-90%), for one model size (1.5B or 7B), n=100 MATH-500 problems, reusing the saved calibration rollouts and layer 15. This is what you run to answer "does steering strength/direction interact with pruning sparsity or pruning method?" |
| [`magnitude_matched_check.py`](magnitude_matched_check.py) | Confound check: the original sweep used an asymmetric alpha grid (`-2, 0, +4`), which confounds "steering toward wrong hurts more" with "the positive alpha is just bigger in magnitude." This adds the missing symmetric points (`+2`, `-4`) so `±2` and `±4` can be compared directly. Supports an optional 3rd CLI arg to restrict which checkpoints run (for splitting work across parallel jobs). |
| [`length_matched_check.py`](length_matched_check.py) | Confound check: wrong rollouts are, on average, longer than correct ones. This rebuilds the direction from only *length-matched* correct/wrong pairs (greedy nearest-length pairing) to rule out "the direction is really just encoding response length," then re-runs the same alpha sweep for direct comparison against the unmatched-direction result. |
| [`truncation_content_analysis.py`](truncation_content_analysis.py) | A different question about the *non-terminating* completions (the ones that hit the token cap instead of stopping): do they contain the correct answer somewhere before the cutoff (and just keep rambling), or does the model genuinely never find it? Also reports how repetitive the tail of the text is and how much self-correction language ("wait", "let me recheck", ...) appears near the end. Consolidates what used to be 4 near-duplicate scripts into one, parametrized by checkpoint + alpha list. |

## Example invocations

```bash
# Step 0 (only if you don't already have calibration rollouts saved)
python save_calib_rollouts.py

# The Llama-8B single-model pilot (no CLI args -- edit the constants at the top to change settings)
python causal_steering.py

# Main sweep: 1.5B model, three alpha values, across all 19 checkpoints (dense + 9 OBC + 9 C4)
python sparsity_sweep.py 1.5B -2,0,2

# Magnitude-matched confound check, 7B model, just the +2 point
python magnitude_matched_check.py 7B 2

# Magnitude-matched confound check, restricted to a subset of checkpoints (for parallel jobs)
python magnitude_matched_check.py 7B 2,-2 dense,obc_50,c4_50

# Length-matched direction ablation, 1.5B model
python length_matched_check.py 1.5B

# Truncation content analysis: dense model at alpha 0 and +2, OBC 50% checkpoint at -2/0/+2
python truncation_content_analysis.py 1.5B dense:0,dense:2,50:-2,50:0,50:2
```

## Output format

The sweep scripts (`sparsity_sweep.py`, `magnitude_matched_check.py`, `length_matched_check.py`)
print per-checkpoint accuracy to stdout as they go — redirect to a log file and parse it, or add a
JSON-dump call if you want structured output. `truncation_content_analysis.py` already writes one
JSON file per setting to `./truncation_analysis/<setting_tag>.json`, containing every completion's
full text plus its truncation/correctness/repetitiveness metrics, so you can go back and read
individual examples.

## What's *not* here

- The actual pruning code that produces the checkpoints these scripts steer — see the top-level
  [`../README.md`](../README.md).
- Any downstream plotting/figure-generation code — this folder only covers running the
  experiments and producing raw numbers/logs, not turning them into figures.
