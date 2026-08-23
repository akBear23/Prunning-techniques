# C4-Calibrated Pruning Code

This folder collects every piece of code used to prune the DeepSeek-R1-Distill / Qwen3 / Llama
reasoning models using **C4** (a generic web-text corpus) as the calibration data — as opposed to
this project's main **OBC-Prune/DAOC** condition, which calibrates from the model's own
correct/wrong reasoning rollouts instead of generic text.

The code originally lives across **two separate repositories** on the shared H100 cluster:

1. `/mnt/data/lannth/COMP/RAC/RAC/open-r1-main` — the main project repo (most of this folder).
2. `/mnt/data/vhoangth2/repos/OBC` — a collaborator's separate repo, which contributed the
   WANDA/ALPS C4 backend script (`external_obc_repo/run_c4_wanda_alps.py`).

Directory structure below mirrors each file's **original relative path** in its source repo, so
that the relative imports inside `run_c4_*.py` (e.g. `from trl.pruner.pruning import ...`) keep
working if you drop `src/` onto your `PYTHONPATH`.

## What actually prunes with C4 — three backends, three entry points

| Backend | Script | Notes |
|---|---|---|
| **SparseGPT** | [`run_c4_baseline.py`](run_c4_baseline.py) | The main C4 baseline. Loads `allenai/c4` (en, streaming), samples `n_calib_samples` chunks of `calib_seq_len` tokens, calls `sparsegpt_prune()`. |
| **ALPS** (ADMM sparse least-squares) | [`run_c4_alps.py`](run_c4_alps.py) | Same recipe as the SparseGPT script, swaps in `alps_prune()`. Written as "the ALPS-backend counterpart to the C4 baseline." |
| **WANDA** and **ALPS** | [`external_obc_repo/run_c4_wanda_alps.py`](external_obc_repo/run_c4_wanda_alps.py) | Lives in the collaborator's separate `OBC` repo, not `open-r1-main`. Takes `--backend {wanda,alps}`. WANDA is fully self-contained in this file (its own `prune_wanda_c4()`); the `alps` branch imports `ALPS_prune` from the same shared library below. |

All three load C4 the same way: `datasets.load_dataset("allenai/c4", "en", split="train",
streaming=True)`, shuffled with a fixed seed, truncated/packed into fixed-length token chunks.

### Example invocations

```bash
# SparseGPT, 1.5B model, 40% sparsity
python run_c4_baseline.py \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --sparsity 0.4 --output_dir ./models/c4_1.5B_sparse40 \
    --n_calib_samples 128 --calib_seq_len 2048 --device auto --dtype bfloat16 --memory_limit_gb 30

# ALPS, same model/sparsity
python run_c4_alps.py \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --sparsity 0.4 --output_dir ./models/c4_alps_1.5B_sparse40

# WANDA (from the external OBC repo script)
python external_obc_repo/run_c4_wanda_alps.py \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --sparsity 0.4 --backend wanda --output_dir ./models/c4_wanda_1.5B_sparse40
```

## Shared pruning library (`src/open_r1/open_r1_trl/trl/`)

All three backends above ultimately call into one shared library, which is a **fork of
HuggingFace's `trl` package with custom pruning code added on top**:

- [`trl/pruner/pruning.py`](src/open_r1/open_r1_trl/trl/pruner/pruning.py) — the actual custom
  contribution. Implements `sparsegpt_prune()`, `prune_wanda()`, `ALPS_prune`/`alps_prune()`,
  `magnitude_prune_layerwise()`, `compute_sparsity()`, and the generic calibration-loader
  machinery (`make_calib_loader`, `prepare_calibration_input`). This is the file to read first if
  you want to understand the actual pruning math.
- [`trl/sparsegpt/`](src/open_r1/open_r1_trl/trl/sparsegpt/) — a vendored copy of the reference
  [SparseGPT](https://github.com/IST-DASLab/sparsegpt) implementation. Only `sparsegpt.py` (the
  `SparseGPT` class) and `quant.py` are actually imported by `pruning.py`; `llama.py`, `opt.py`,
  `bloom.py`, `modelutils.py`, `datautils.py` are the original repo's model-specific reference
  scripts, kept for context but not on the C4 pruning code path.
- [`trl/data_utils.py`](src/open_r1/open_r1_trl/trl/data_utils.py), [`trl/__init__.py`](src/open_r1/open_r1_trl/trl/__init__.py), [`trl/import_utils.py`](src/open_r1/open_r1_trl/trl/import_utils.py) —
  stock/unmodified `trl` package files, included only because `pruning.py` needs
  `maybe_apply_chat_template` from `data_utils.py`, and importing `trl.pruner.pruning` first runs
  `trl/__init__.py`. **These are not part of this project's custom work** — in a real install you
  would just `pip install trl` (this repo pins `__version__ = "0.19.0.dev0"`) rather than vendor
  them, but they're included here so the folder actually runs standalone without a full TRL
  checkout.

The rest of the original `trl` package (`trainer/`, `environment/`, `extras/`, `models/`,
`examples/`, `tests/`, etc.) is **not** included — none of it is on the C4 pruning path. If you
need it, `pip install trl` gives you the genuine upstream package; this folder only ships the
custom pruner/sparsegpt additions plus the three small files needed to import them cleanly.

## Orchestration wrappers (`scripts/` and top-level `run_c4_*_h100.py` / `run_c4_*_baseline.py`)

These call the entry-point scripts above in a loop across sparsity levels / model sizes, run the
downstream MATH-500 evaluation via `lighteval`, and skip work that's already been done (checking
for existing output files). They're cluster-specific (hardcoded paths, conda env names) and meant
as **documentation of exactly how each C4 checkpoint was produced**, not as portable code:

- [`run_c4_baseline_h100.py`](run_c4_baseline_h100.py) — SparseGPT+C4, 1.5B @ 40%/50%, then evals.
- [`run_c4_full_baseline.py`](run_c4_full_baseline.py) — broader SparseGPT+C4 sweep driver.
- [`run_c4_14b_retry.py`](run_c4_14b_retry.py) — retry/resume driver for the 14B model specifically.
- [`scripts/run_wanda_baselines_gpu82.py`](scripts/run_wanda_baselines_gpu82.py) — drives
  `external_obc_repo/run_c4_wanda_alps.py` (and the SSGR condition) for 1.5B/7B @ 40%/50%, plus
  evaluation.
- [`scripts/run_wanda_14b.py`](scripts/run_wanda_14b.py) — the 14B-specific WANDA driver.

## Activation steering on the pruned checkpoints (`steering/`)

[`steering/`](steering/) contains a separate but related set of experiments: given a pruned
checkpoint from the sweep above, does injecting a "correct vs. wrong reasoning" direction into a
decoder layer's hidden state during generation *causally* change accuracy, and does that effect
depend on sparsity or pruning method (OBC-Prune vs. C4)? See
[`steering/README.md`](steering/README.md) for the method, prerequisites, and example commands —
start there if you want to run these.

## What's deliberately *not* here

- Evaluation/scoring code (`lighteval` invocations, results parsing) — referenced by the
  orchestration wrappers above but not itself pruning code, so out of scope for this folder.
- The **OBC-Prune/DAOC** calibration pipeline (`src/open_r1/daoc_prune/`) — a completely separate
  condition that calibrates from the model's own rollouts rather than C4 text. Not part of "prune
  with C4" by definition.
- **SSGR** and **RAC** conditions — other calibration methods in the same project, out of scope
  here for the same reason.

## Setup notes

- Requires `torch`, `transformers`, `datasets` (for streaming `allenai/c4`), plus whatever
  `trl/__init__.py`'s lazy-import machinery needs at runtime (it defers loading, so importing
  `trl.pruner.pruning` directly is cheap and doesn't require the full upstream `trl` install).
- `run_c4_baseline.py` / `run_c4_alps.py` expect to be run from a working directory where `trl` on
  `sys.path` resolves to `src/open_r1/open_r1_trl/trl` in this folder (e.g. add `src/open_r1/open_r1_trl`
  to `PYTHONPATH`), matching how they're invoked in `open-r1-main` originally.
- `external_obc_repo/run_c4_wanda_alps.py`'s `--backend alps` path has the same `trl.pruner.pruning`
  dependency; its `--backend wanda` path is fully self-contained (no `trl` import needed).
