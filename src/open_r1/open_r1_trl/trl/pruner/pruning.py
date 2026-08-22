"""
SparseGPT pruning with **budget‑aware grouping**.

We instantiate as many SparseGPT pruners in parallel as will fit inside
`memory_limit_gb` (default = 60).  Groups are processed sequentially:

  ┌─ group 1 – collect stats (hooks live) ─┐
  │  prune L1, free  │  prune L2, free …   │
  └─────────────────────────────────────────┘
  ┌─ group 2 – same pattern … ─────────────┘
"""

from __future__ import annotations
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import PreTrainedTokenizerBase
import transformers
from typing import Iterable, Tuple
from ..sparsegpt.sparsegpt import SparseGPT
from ..data_utils import maybe_apply_chat_template

# ──────────────────────────────────────────────────────────────────────────────
# helper utils (unchanged from upstream, kept verbatim)
# ──────────────────────────────────────────────────────────────────────────────
def _count_loader_tokens(loader: DataLoader) -> int:
    n = 0
    for batch in loader:
        n += batch["input_ids"].numel()
    return n


def _row_to_prompt(
    row: dict,
    tokenizer: PreTrainedTokenizerBase,
    prompt_column: str = "prompt",
) -> str:
    if isinstance(row.get(prompt_column), str):
        return row[prompt_column]
    if prompt_column in row:
        return maybe_apply_chat_template(row, tokenizer)["prompt"]
    if "input_ids" in row:
        return tokenizer.decode(row["input_ids"], skip_special_tokens=True)
    if "text" in row:
        return row["text"]
    raise KeyError(
        f"Cannot find '{prompt_column}', 'input_ids', or 'text' in row – "
        "please check your dataset or --dataset_prompt_column."
    )


def make_calib_loader(
    dataset,
    tokenizer: PreTrainedTokenizerBase,
    tokens: int,
    batch_size: int = 8,
    *,
    prompt_column: str = "prompt",
    weight_col: str | None = None,
) -> DataLoader:
    n_tok, prompts, weights = 0, [], []
    for row in dataset:
        p = _row_to_prompt(row, tokenizer, prompt_column)
        n_tok += len(tokenizer(p).input_ids)
        prompts.append(p)
        w = float(row.get(weight_col, 1.0)) if weight_col else 1.0
        weights.append(w)
        if n_tok >= tokens:
            break

    def _collate(batch_prompts: list[str]):
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )
        idx = [prompts.index(p) for p in batch_prompts]
        enc["weights"] = torch.tensor([weights[i] for i in idx], dtype=torch.float32)
        return enc

    if weight_col is None:
        return DataLoader(prompts, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(prompts, batch_size=batch_size, sampler=sampler, collate_fn=_collate)

# ──────────────────────────────────────────────────────────────────────────────
#  SparseGPT – budget‑aware grouping
# ──────────────────────────────────────────────────────────────────────────────
def _is_mlp(name: str) -> bool:
    kw = (
        "mlp",
        "ff",
        "feed_forward",
        "ffn",
        "dense_h_to_4h",
        "gate_proj",
        "down_proj",
        "up_proj",
    )
    return any(k in name.lower() for k in kw)


def _estimate_hessian_gb(layer: nn.Module) -> float:
    """Rough fp32 size of the Hessian for *layer* in **gigabytes**."""
    if isinstance(layer, nn.Conv2d):
        cols = layer.weight.data.flatten(1).shape[1]
    elif isinstance(layer, transformers.Conv1D):
        cols = layer.weight.data.t().shape[1]
    else:  # nn.Linear
        cols = layer.weight.data.shape[1]
    bytes_ = cols * cols * 4  # float32
    return bytes_ / (1024**3)  # GiB


def sparsegpt_prune(
    model: nn.Module,
    calib_loader: DataLoader,
    sparsity: float,
    *,
    prunen: int | None = None,
    prunem: int | None = None,
    device: str = "cuda",
    scope: str = "all",
    memory_limit_gb: float = 30.0,
    thirds_to_prune: Tuple[int, ...] = (1, 2, 3),
) -> None:
    """
    Prune with SparseGPT, grouping layers so that the sum of their Hessian
    footprints never exceeds memory limit.  Everything stays in float32.
    """
    PRUNE_TYPES = (nn.Linear, nn.Conv2d, transformers.Conv1D)

    layers: list[tuple[str, nn.Module]] = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, PRUNE_TYPES)
        and m.weight.requires_grad
        and (scope == "all" or _is_mlp(n))
    ]

    layers_before = len(layers)
    layers = _subset_by_thirds(layers, thirds_to_prune)
    print(
        f"[SparseGPT] pruning thirds {sorted(set(thirds_to_prune))} → "
        f"{len(layers)}/{layers_before} layers selected"
    )

    print(f"[SparseGPT] total layers eligible: {len(layers)}")

    group: list[tuple[str, nn.Module]] = []
    group_mem, total_done = 0.0, 0
    label = f"{prunen}:{prunem}" if prunen and prunem else f"{sparsity*100:.1f}%"

    def _process_group(group_layers: list[tuple[str, nn.Module]], idx0: int) -> None:
        nonlocal total_done
        if not group_layers:
            return

        print(
            f"[SparseGPT] -- processing group {idx0}‑{idx0+len(group_layers)-1} "
            f"(H memory {group_mem:.2f} GB) --"
        )

        # 1) create pruners and hooks
        pruners: dict[str, SparseGPT] = {}
        hooks: list[torch.utils.hooks.RemovableHandle] = []

        def _make_hook(name: str):
            pr = pruners[name]

            def _hook(mod, inp, out, **kw):
                pr.add_batch(inp[0].detach(), out.detach(), weights=kw.get("weights"))
            return _hook

        for name, lyr in group_layers:
            pruners[name] = SparseGPT(lyr)
            hooks.append(lyr.register_forward_hook(_make_hook(name)))

        # 2) run calibration forward (single pass)
        with torch.inference_mode():
            for batch in calib_loader:
                tgt_dev = (
                    next(iter(model.hf_device_map.values()))
                    if hasattr(model, "hf_device_map")
                    else device
                )
                batch = {k: v.to(tgt_dev) for k, v in batch.items()}
                model(**batch)

        for h in hooks:
            h.remove()

        # 3) prune each layer sequentially inside the group
        for name, _ in group_layers:
            pr = pruners[name]
            pr.fasterprune(
                sparsity,
                prunen=(prunen or 0),
                prunem=(prunem or 0),
            )
            pr.free()
            del pruners[name]
            torch.cuda.empty_cache()
            total_done += 1
            print(f"[SparseGPT] layer {total_done}/{len(layers)} pruned (target {label})")

        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # main loop – build groups under memory budget
    # ------------------------------------------------------------------ #
    for idx, (lname, layer) in enumerate(layers):
        mem = _estimate_hessian_gb(layer) * 1.15  # 15 % overhead safety
        if mem > memory_limit_gb:                 # pathological single layer
            print(
                f"[SparseGPT] WARNING: single layer '{lname}' "
                f"requires {mem:.2f} GB > budget {memory_limit_gb} GB; "
                "handling it alone."
            )
            if group:  # flush current group first
                _process_group(group, idx - len(group))
                group, group_mem = [], 0.0
            _process_group([(lname, layer)], idx)
            continue

        if group_mem + mem <= memory_limit_gb:
            group.append((lname, layer))
            group_mem += mem
        else:                                     # flush and start new group
            _process_group(group, idx - len(group))
            group, group_mem = [(lname, layer)], mem

    _process_group(group, len(layers) - len(group))

    if hasattr(model, "fuse"):
        model.fuse()

    realised = compute_sparsity(model)
    print(f"[SparseGPT] realised sparsity: {realised*100:.2f}% (budget {memory_limit_gb} GB)")


def compute_sparsity(model: nn.Module) -> float:
    """Return the fraction (0‑1) of *all* parameters that are exactly zero."""
    total, zeros = 0, 0
    with torch.no_grad():
        for p in model.parameters():
            total += p.numel()
            zeros += (p == 0).sum().item()
    return zeros / total if total > 0 else 0.0

def magnitude_prune_layerwise(
    model: torch.nn.Module,
    sparsity: float,
    device: str = "cuda",
) -> float:
    """
    Unstructured magnitude pruning applied *independently per layer*.
    """
    assert 0.0 <= sparsity < 1.0, "sparsity must be in [0, 1)"
    model.to(device).eval()

    PRUNE_TYPES = (nn.Linear, nn.Conv2d, transformers.Conv1D)

    with torch.no_grad():
        for module in model.modules():
            if not (isinstance(module, PRUNE_TYPES) and module.weight.requires_grad):
                continue

            w = module.weight.detach()
            k = int(w.numel() * sparsity)
            if k == 0:
                continue

            th = w.abs().flatten().kthvalue(k).values.item()
            w[w.abs() <= th] = 0.0

    torch.cuda.empty_cache()
    realised = compute_sparsity(model)
    print(f"[Mag-Layer] target {sparsity * 100:.1f}% → realised {realised * 100:.2f}%")
    return realised


# ─────────────────────── imports ───────────────────────
from typing import Any
import torch, torch.nn as nn
from transformers.tokenization_utils_base import BatchEncoding

# ───────────────────── helper: locate blocks ───────────
def _get_decoder_layers(model: nn.Module):
    """
    Return the transformer block stack independent of model layout.
    Works for OPT/GPT-J (model.model.decoder.layers) and
    Llama/Qwen-2/Gemma (model.model.layers).
    """
    core = getattr(model, "model", model)
    return getattr(core, "decoder", core).layers

# ───────────────────── helper: seq length ──────────────
def _batch_seq_len(batch: Any) -> int:
    """
    Token length of the first sample in *batch*.

    Accepts:
        • tuple/list            -> (input_ids, …)
        • BatchEncoding / dict  -> {'input_ids': …}
        • torch.Tensor          -> ids
        • tokenizers.Encoding   -> .ids
    """
    if isinstance(batch, (tuple, list)):
        batch = batch[0]

    if hasattr(batch, "keys") and "input_ids" in batch:
        item = batch["input_ids"]
        return item.shape[-1] if torch.is_tensor(item) else len(item[0])

    if torch.is_tensor(batch):
        return batch.shape[-1]

    if hasattr(batch, "ids"):                 # tokenizers.Encoding
        return len(batch.ids)

    raise TypeError(f"Unsupported batch type: {type(batch)}")

# ───────────────────── calibration collector ───────────

def prepare_calibration_input(model, loader, device="cuda"):
    DTYPE       = next(model.parameters()).dtype
    hidden      = model.config.hidden_size
    max_seq     = max(b["input_ids"].shape[1] for b in loader)
    nsamples    = len(loader.dataset)

    inps = torch.empty(nsamples, max_seq, hidden, dtype=DTYPE, device="cpu")
    outs = torch.empty_like(inps)
    cache = {"i": 0}

    layers = _get_decoder_layers(model)
    orig0  = layers[0]
    layers[0] = Catcher(orig0, inps, cache)

    try:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)
    except ValueError:
        pass

    layers[0] = orig0
    return inps, outs, None, None

# ───────────────────── WANDA pruning ╌ layer-wise ──────
def prune_wanda(model: nn.Module,
                calib_loader,
                sparsity: float,
                device: str = "cuda") -> None:
    """
    Layer-wise unstructured WANDA pruning.

    Key change: per-layer tensors (inps/outs/pos_ids/mask and rotary cache)
    are always moved to the device of the *layer itself*, derived via
    `next(layer.parameters()).device`. This avoids CPU↔CUDA mismatches that
    crash fused Liger RMSNorm/Triton kernels.
    """
    # ------------------------------------------------------------------ #
    # 0. setup
    # ------------------------------------------------------------------ #
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    PARAM0 = next(model.parameters())
    DTYPE  = PARAM0.dtype
    MODEL_DEV = PARAM0.device

    # ------------------------------------------------------------------ #
    # 1. collect calibration activations
    # ------------------------------------------------------------------ #
    inps, outs, attn_mask, _ = prepare_calibration_input(
        model, calib_loader, device
    )  # inps/outs come back on CPU
    inps, outs = inps.to(DTYPE), outs.to(DTYPE)
    nsamp, seq_len, _ = inps.shape
    layers = _get_decoder_layers(model)

    # template position ids (will be moved per-layer)
    pos_ids_t = torch.arange(seq_len, dtype=torch.long, device="cpu").unsqueeze(0)

    # the module that exposes rotary_emb on Qwen2/Llama-style models
    base_model = getattr(model, "model", model)

    # cache: device -> all-ones mask to avoid re-allocations
    _full_mask_cache: dict[torch.device, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # 2. iterate over transformer blocks
    # ------------------------------------------------------------------ #
    for i, layer in enumerate(layers):
        subset = find_layers(layer)           # {name: sub-module}

        # === robust, always-correct device for THIS layer ===
        layer_dev: torch.device = next(layer.parameters()).device

        # move our rolling buffers & ids to this layer's device
        inps  = inps.to(layer_dev, non_blocking=True)
        outs  = outs.to(layer_dev, non_blocking=True)
        pos_ids = pos_ids_t.to(layer_dev, non_blocking=True)

        # -------- ensure we always pass a legal attention-mask -----------
        if attn_mask is None:
            mask = _full_mask_cache.get(layer_dev)
            if mask is None or mask.shape[1] < seq_len:
                mask = torch.ones(1, seq_len, dtype=torch.bool, device=layer_dev)
                _full_mask_cache[layer_dev] = mask
        else:
            mask = attn_mask.to(layer_dev, non_blocking=True)
        # ----------------------------------------------------------------

        # -------- rotary position embeddings (Qwen-2 / Llama) -----------
        extra_kw = {}
        if hasattr(base_model, "rotary_emb"):
            cache_name = "_wanda_pos_emb"
            cached = getattr(base_model, cache_name, None)
            needs_new = (
                cached is None
                or cached[0].device != layer_dev
                or cached[0].shape[1] < seq_len
            )
            if needs_new:
                dummy = torch.zeros(
                    1, seq_len, base_model.config.hidden_size,
                    dtype=DTYPE, device=layer_dev
                )
                setattr(base_model, cache_name,
                        base_model.rotary_emb(dummy, pos_ids))
            extra_kw["position_embeddings"] = getattr(base_model, cache_name)
        # ----------------------------------------------------------------

        # 2a. attach forward hooks to accumulate row scalers
        wrappers = {n: WrappedGPT(m) for n, m in subset.items()}
        hooks = [
            subset[n].register_forward_hook(
                lambda m, inp, out, name=n:
                    wrappers[name].add_batch(inp[0].data, out.data))
            for n in wrappers
        ]

        # 2b. run each calibration slice through *this* block only
        for j in range(nsamp):
            layer(
                inps[j:j+1],
                attention_mask=mask,
                position_ids=pos_ids,
                **extra_kw
            )[0]

        for h in hooks:
            h.remove()

        # ------------------------------------------------------------------
        # 3. compute WANDA metric & apply mask
        # ------------------------------------------------------------------
        for name, sub in subset.items():
            W = sub.weight.data
            # scaler_row was accumulated on sub.weight.device already
            scaler = torch.sqrt(wrappers[name].scaler_row.reshape(1, -1).to(W.device))
            metric = torch.abs(W) * scaler

            # --- robust k computation ------------------------------------
            n_col = metric.size(1)
            k = int(round(n_col * sparsity))      # round instead of floor
            if k == 0:
                continue                          # nothing to prune
            if k >= n_col:
                k = n_col - 1                     # keep at least one column
            # --------------------------------------------------------------

            idx = torch.topk(metric, k, dim=1, largest=False, sorted=False).indices
            W.scatter_(1, idx, 0)

            print(f"[WANDA] pruned layer {i} – {name} ({k}/{n_col} per row)")

        # ------------------------------------------------------------------
        # 4. propagate activations to serve next block
        # ------------------------------------------------------------------
        for j in range(nsamp):
            outs[j] = layer(
                inps[j:j+1],
                attention_mask=mask,
                position_ids=pos_ids,
                **extra_kw
            )[0].detach()
        inps, outs = outs, inps  # swap buffers

    # ------------------------------------------------------------------ #
    # 5. restore config & clean up
    # ------------------------------------------------------------------ #
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()





class Catcher(nn.Module):
    """
    Capture the hidden-state that enters the **first** decoder block during
    calibration.  We copy each sample into a pre-allocated CPU buffer `inps`
    (shape = [N, max_seq, H]), truncating or padding with zeros so every row
    fits exactly `max_seq` tokens.
    """
    def __init__(self,
                 inner: nn.Module,
                 inps_buf: torch.Tensor,
                 cache: dict):
        super().__init__()
        self.inner  = inner          # the real decoder layer
        self.inps   = inps_buf       # (N, max_seq, H) CPU buffer
        self.cache  = cache          # {'i': row_idx, …}

        # propagate attributes expected elsewhere (e.g. attention_type)
        for attr in ("attention_type",):
            if hasattr(inner, attr):
                setattr(self, attr, getattr(inner, attr))

    # ------------------------------------------------------------------ #
    # forward: copy -> raise ValueError to abort the full forward pass
    # ------------------------------------------------------------------ #
    def forward(self, x, **kw):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, S, H).  We only need the first sequence in the batch.
        """
        # --- pick first sequence and move to CPU ---
        seq_cpu = x[0].detach().cpu()        # (S, H)
        seq_len, hidden = seq_cpu.shape

        # --- locate destination row in the buffer ---
        dest     = self.inps[self.cache["i"]]   # (max_seq, H)
        max_len  = dest.size(0)                 # buffer's token capacity
        dest.zero_()                            # clear previous contents

        # --- copy (truncate if too long) --------------------------------
        n = min(seq_len, max_len)
        dest[:n].copy_(seq_cpu[:n])

        # --- book-keeping ----------------------------------------------
        self.cache["i"] += 1
        self.cache["attention_mask"] = None     # masks no longer valid
        self.cache["position_ids"]   = None

        # stop the full model forward; we only wanted the hidden state
        raise ValueError

    # delegate all other attributes/methods to the wrapped layer
    def __getattr__(self, name):
        return getattr(self.inner, name)


def find_layers(module: nn.Module,
                layer_types: tuple[type[nn.Module], ...] = (nn.Linear,)) -> dict[str, nn.Module]:
    """
    Return every sub-module in *module* whose type is in *layer_types*.
    The keys are fully-qualified names (as in `named_modules()`);
    the values are the actual sub-module objects.

    Default behaviour: collect **all nn.Linear** layers.
    Extend *layer_types* if you also want nn.Conv1d, nn.Conv2d, etc.
    """
    result: dict[str, nn.Module] = {}
    for name, sub in module.named_modules():
        if isinstance(sub, layer_types):
            result[name] = sub
    return result

class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out, weights=None):
        """
        weights: optional (n_tokens,) per-token importance weights (e.g. DAOC's
        attention-suppression weights). Applied as sqrt(w) on each token's input
        vector before the squared-norm accumulation, so a token with weight w
        contributes w * x^2 to scaler_row instead of x^2 — the same "scale by
        sqrt(w) before the outer product" trick SparseGPT.add_batch uses for its
        Hessian, since scaler_row is just diag(H) for a plain (unweighted) input.
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        if weights is not None:
            w = weights.to(inp.device, dtype=torch.float32).reshape(1, -1)
            inp = inp * w.sqrt()
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples


def _subset_by_thirds(items: list, thirds: Iterable[int]) -> list:
    """
    Split `items` into 3 contiguous thirds and keep only those indices in `thirds`.
    Third indices are 1-based: 1 = first third, 2 = second, 3 = last third.
    """
    n = len(items)
    if n == 0:
        return items
    a, b = n // 3, (2 * n) // 3
    buckets = [items[:a], items[a:b], items[b:]]  # 3 contiguous slices
    keep = set(int(t) for t in thirds if t in (1, 2, 3))
    out: list = []
    for i, bucket in enumerate(buckets, start=1):
        if i in keep:
            out.extend(bucket)
    return out


# ──────────────────────────────────────────────────────────────────────────────
#  ALPS – ADMM-based sparse least-squares pruning
#  Vendored from https://github.com/mazumder-lab/ALPS (alps.py: ALPS_prune),
#  add_batch patched to accept optional per-token `weights` (unused by the
#  plain alps_prune() below — RAC calls it with weights=None, same as
#  SparseGPT/WANDA above; the DAOC-weighted caller lives in daoc_prune).
# ──────────────────────────────────────────────────────────────────────────────

class ALPS_prune:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device

        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        self.XtX = torch.zeros((self.columns, self.columns), device=self.dev).float()
        self.nsamples = 0

    def add_batch(self, inp, out, weights=None):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
            if len(out.shape) == 3:
                out = out.reshape((-1, out.shape[-1]))
        out = out.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        inp = inp.float()
        out = out.float()  # unused downstream — YtX is derived from the frozen dense
        # weight in ALPS_admm (Y = W_dense @ X), not accumulated here; see ALPS_admm.

        if weights is not None:
            w = weights.to(inp.device, dtype=torch.float32).reshape(1, -1)
            inp = inp * w.sqrt()

        self.XtX += inp.matmul(inp.t())
        self.nsamples += tmp

    def ALPS_admm(self, sp, nm_n=0, nm_m=0, rho=0.1, max_iter=300, update_iter=3, switch_iter=30):
        dev = str(self.dev)
        W = self.layer.weight.data.clone()
        W = W.float()
        W = W.to(dev)
        # NOTE: deviates from upstream, which forces self.XtX to CPU here and
        # runs the eigendecomposition below on CPU LAPACK. XtX is already
        # GPU-resident (accumulated on-device in add_batch), and torch.linalg.eigh
        # supports CUDA tensors directly via cuSOLVER — for the columns sizes
        # here (a few thousand to ~19k) that's dramatically faster than CPU
        # LAPACK and was the dominant per-layer bottleneck (multi-minute CPU
        # eigh on an otherwise-idle GPU). Keeping XtX on `dev` throughout is
        # mathematically identical; only which device performs the linear
        # algebra changes.

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        damp1 = 0.01 * torch.mean(torch.diag(self.XtX)).item()
        diag = torch.arange(self.XtX.shape[0], device=self.XtX.device)
        self.XtX[diag, diag] += damp1

        X_norm = torch.diag(self.XtX).sqrt() + 1e-8
        self.XtX = self.XtX / X_norm
        self.XtX = (self.XtX.T / X_norm).T

        self.YtX = torch.zeros_like(W)
        self.YtX = torch.matmul(W * X_norm, self.XtX).to(dev)

        B = (W * X_norm.to(dev)).t().clone()
        W = None
        B_orig = B.cpu().clone()
        V = torch.zeros_like(B)
        D = torch.zeros_like(B)
        D_suppp = torch.zeros_like(B)
        D_supp = torch.zeros_like(B)

        totp, num_cout = B.shape
        L, Q = torch.linalg.eigh(self.XtX.double())
        XTX_inv = (Q @ ((1 / (L + rho)) * Q).T).float().to(dev)

        init_rho = False
        fix_supp = False
        D_fix = torch.zeros_like(D)

        Res0 = self.YtX.T.cpu()
        Res0 = torch.sum(B_orig.cpu() * Res0)
        Res0 = torch.sum(Res0)

        params = B.shape[0] * B.shape[1]
        k_spar = int(np.round((1 - sp) * params))

        if nm_n == 0:
            D = B.clone().reshape(-1)
            _, loss_idx = torch.topk(-D**2, totp * num_cout - k_spar)
            D[loss_idx] = 0
            D_suppp = (D == 0).to(torch.float)
            D = D.reshape(totp, num_cout)
        else:
            new_dim = int(totp * num_cout / nm_m)
            k_spar = totp * num_cout * nm_n / nm_m

            D = B.clone().t().reshape((new_dim, nm_m))
            _, loss_idx = torch.topk(-D**2, nm_m - nm_n, dim=1)
            D = D.scatter(src=torch.zeros((new_dim, nm_m - nm_n)).to(dev), dim=1, index=loss_idx)
            D_suppp = (D == 0).to(torch.float)
            D = D.reshape(num_cout, totp).t()

        D_init = D.clone()
        for i_admm in range(max_iter):
            B = XTX_inv @ (self.YtX.T - V + rho * D)

            if fix_supp:
                D = ((V + rho * B) / rho) * D_fix
            elif nm_n == 0:
                D = ((V + rho * B) / rho).reshape(-1)
                _, loss_idx = torch.topk(-D**2, totp * num_cout - k_spar)
                D[loss_idx] = 0
                D = D.reshape(totp, num_cout)
            else:
                D = ((V + rho * B) / rho).t().reshape((new_dim, nm_m))
                _, loss_idx = torch.topk(-D**2, nm_m - nm_n, dim=1)
                D = D.scatter(src=torch.zeros((new_dim, nm_m - nm_n)).to(dev), dim=1, index=loss_idx)
                D_supp = (D == 0).to(torch.float)
                D = D.reshape(num_cout, totp).t()

            V = V + rho * (B - D)

            if (i_admm + 1) % update_iter == 0:
                if nm_n == 0:
                    D_supp = (D.reshape(-1) == 0).to(torch.float)
                supp_change = torch.sum((D_supp - D_suppp) ** 2)

                if not fix_supp:
                    if supp_change / k_spar > 0.1:
                        init_rho = True
                        rho *= 1.3
                    elif supp_change / k_spar > 0.005:
                        init_rho = True
                        rho *= 1.2
                    elif supp_change > 0.5:
                        if init_rho:
                            rho *= 1.1
                        else:
                            rho /= 5
                            B = B_orig.clone().to(dev)
                            D = D_init.clone().to(dev)
                            V = torch.zeros_like(B).to(dev)
                    else:
                        if init_rho:
                            break
                        else:
                            rho /= 5

                D_suppp = D_supp.clone()
                if rho > 1e6:
                    rho = 1e6

                XTX_inv = (Q @ ((1 / (L + rho)) * Q).T).float().to(dev)

                if nm_n == 0:
                    Btest = B.reshape(-1)
                    _, loss_idx = torch.topk(-Btest**2, totp * num_cout - k_spar)
                    Btest[loss_idx] = 0
                    Btest = Btest.reshape(totp, num_cout)
                else:
                    Btest = B.t().reshape((new_dim, nm_m))
                    _, loss_idx = torch.topk(-Btest**2, nm_m - nm_n, dim=1)
                    Btest = Btest.scatter(src=torch.zeros((new_dim, nm_m - nm_n)).to(dev), dim=1, index=loss_idx)
                    Btest = Btest.reshape(num_cout, totp).t()

                Resc = torch.matmul(self.XtX.to(dev), Btest) - self.YtX.T
                Resc = torch.diag(torch.matmul((Btest - B_orig.to(dev)).t(), Resc))
                errorc = (torch.sum(Resc).to("cpu") / Res0).item()  # noqa: F841 (parity with upstream)

                if i_admm >= switch_iter and supp_change / k_spar < 0.0003:
                    break

        if nm_n == 0:
            B = B.reshape(-1)
            _, loss_idx = torch.topk(-B**2, totp * num_cout - k_spar)
            B[loss_idx] = 0
            B = B.reshape(totp, num_cout)
        else:
            B = B.t().reshape((new_dim, nm_m))
            _, loss_idx = torch.topk(-B**2, nm_m - nm_n, dim=1)
            B = B.scatter(src=torch.zeros((new_dim, nm_m - nm_n)).to(dev), dim=1, index=loss_idx)
            B = B.reshape(num_cout, totp).t()

        V = None
        D = None

        B = self.cg_batch(
            self.XtX.to(dev), self.YtX.T,
            (B != 0).to(torch.float), M_bmm=None, X0=B, rtol=1e-4, atol=0.0, maxiter=10, verbose=False,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if isinstance(self.layer, transformers.Conv1D):
            self.layer.weight.data = (B.t() / X_norm.to(dev)).t().reshape(
                self.layer.weight.shape
            ).to(self.layer.weight.data.dtype)
        else:
            self.layer.weight.data = (B.t() / X_norm.to(dev)).reshape(
                self.layer.weight.shape
            ).to(self.layer.weight.data.dtype)

    # A modified version of https://github.com/sbarratt/torch_cg (vendored via ALPS upstream)
    def cg_batch(self, A, B, A_supp, M_bmm=None, X0=None, rtol=1e-3, atol=0.0, maxiter=None, verbose=False):
        n, m = B.shape

        if M_bmm is None:
            M_bmm = lambda x: x
        if X0 is None:
            X0 = M_bmm(B)
        if maxiter is None:
            maxiter = 5 * n

        X_k = X0
        R_k = B - A @ X_k
        R_k = R_k * A_supp
        Z_k = M_bmm(R_k)
        P_k = torch.zeros_like(Z_k)
        P_k1, R_k1, Z_k1 = P_k, R_k, Z_k

        B_norm = torch.norm(B, dim=1)
        stopping_matrix = torch.max(rtol * B_norm, atol * torch.ones_like(B_norm))

        for k in range(1, maxiter + 1):
            Z_k = M_bmm(R_k)
            if k == 1:
                P_k = Z_k
                R_k1, X_k1, Z_k1 = R_k, X_k, Z_k
            else:
                R_k2, Z_k2, P_k1 = R_k1, Z_k1, P_k
                R_k1, Z_k1, X_k1 = R_k, Z_k, X_k
                denominator = (R_k2 * Z_k2).sum(0)
                denominator[denominator == 0] = 1e-8
                beta = (R_k1 * Z_k1).sum(0) / denominator
                P_k = Z_k1 + beta.unsqueeze(0) * P_k1

            denominator = (P_k * (A @ P_k)).sum(0)
            denominator[denominator == 0] = 1e-8
            alpha = (R_k1 * Z_k1).sum(0) / denominator
            X_k = X_k1 + alpha.unsqueeze(0) * P_k
            R_k = R_k1 - alpha.unsqueeze(0) * (A @ P_k)
            R_k = R_k * A_supp

            residual_norm = torch.norm(A @ X_k - B, dim=1)
            if (residual_norm <= stopping_matrix).all():
                break

        return X_k

    def free(self):
        self.XtX = None
        self.YtX = None
        torch.cuda.empty_cache()


def _estimate_alps_gb(layer: nn.Module) -> float:
    """Rough fp32 size of ALPS's XtX for *layer*, in GiB (same shape as SparseGPT's H)."""
    if isinstance(layer, nn.Conv2d):
        cols = layer.weight.data.flatten(1).shape[1]
    elif isinstance(layer, transformers.Conv1D):
        cols = layer.weight.data.t().shape[1]
    else:
        cols = layer.weight.data.shape[1]
    return (cols * cols * 4) / (1024**3)


def alps_prune(
    model: nn.Module,
    calib_loader: DataLoader,
    sparsity: float,
    *,
    prunen: int = 0,
    prunem: int = 0,
    device: str = "cuda",
    scope: str = "all",
    memory_limit_gb: float = 30.0,
    thirds_to_prune: Tuple[int, ...] = (1, 2, 3),
    rho: float = 0.1,
    max_iter: int = 300,
) -> None:
    """
    Prune with ALPS (ADMM sparse least-squares), same budget-aware layer
    grouping as sparsegpt_prune above.

    `lm_head` is excluded: ALPS's unstructured (prunen=0) branch does a
    *global* flatten-and-topk over the whole (columns * rows) weight matrix,
    unlike WANDA's per-row topk or SparseGPT's column-blocked update — both
    of which stay cheap regardless of `rows`. For lm_head, rows=vocab_size
    (~150k), so that flatten is ~230M elements and torch.topk's CUDA temp
    buffers for a huge-fraction selection can exceed GPU memory (observed:
    an 86 GiB allocation request on a 1.5B model). Every other backend here
    still prunes it.
    """
    PRUNE_TYPES = (nn.Linear, nn.Conv2d, transformers.Conv1D)

    layers: list[tuple[str, nn.Module]] = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, PRUNE_TYPES)
        and m.weight.requires_grad
        and (scope == "all" or _is_mlp(n))
    ]
    layers_before = len(layers)
    layers = _subset_by_thirds(layers, thirds_to_prune)

    if not prunen:  # prunen may be None (GRPOConfig's prune_N default) as well as 0
        lm_head_layers = [n for n, _ in layers if n.endswith("lm_head")]
        for n in lm_head_layers:
            print(f"[ALPS] excluding '{n}' — global unstructured top-k doesn't fit vocab-sized output")
        layers = [(n, m) for n, m in layers if not n.endswith("lm_head")]

    print(
        f"[ALPS] pruning thirds {sorted(set(thirds_to_prune))} → "
        f"{len(layers)}/{layers_before} layers selected"
    )

    if torch.cuda.is_available():
        # Larger buffer than SparseGPT/WANDA need: XtX stays GPU-resident
        # throughout ALPS_admm (see class docstring), so whichever single layer
        # is currently being solved adds a transient double-precision copy of
        # its XtX plus eigh's Q output and the B/V/D ADMM iterates on top of
        # the group's static float32 XtX total. Observed: a several-GiB
        # overshoot with only a 2 GB buffer on a 14B-scale layer.
        free_gb = (torch.cuda.get_device_properties(0).total_memory
                   - torch.cuda.memory_reserved()) / 1024**3
        safe_limit = max(1.0, free_gb - 15.0)
        if memory_limit_gb > safe_limit:
            print(
                f"[ALPS] memory_limit_gb={memory_limit_gb:.1f} exceeds safe budget "
                f"({safe_limit:.1f} GB free − 15 GB buffer); capping to {safe_limit:.1f} GB"
            )
            memory_limit_gb = safe_limit

    group: list[tuple[str, nn.Module]] = []
    group_mem, total_done = 0.0, 0

    def _process_group(group_layers: list[tuple[str, nn.Module]], idx0: int) -> None:
        nonlocal total_done
        if not group_layers:
            return

        print(f"[ALPS] -- processing group {idx0}-{idx0+len(group_layers)-1} (XtX memory {group_mem:.2f} GB) --")

        pruners: dict[str, ALPS_prune] = {}
        hooks: list[torch.utils.hooks.RemovableHandle] = []

        def _make_hook(name: str):
            pr = pruners[name]

            def _hook(mod, inp, out, **kw):
                pr.add_batch(inp[0].detach(), out.detach(), weights=kw.get("weights"))
            return _hook

        for name, lyr in group_layers:
            pruners[name] = ALPS_prune(lyr)
            hooks.append(lyr.register_forward_hook(_make_hook(name)))

        with torch.inference_mode():
            for batch in calib_loader:
                tgt_dev = (
                    next(iter(model.hf_device_map.values()))
                    if hasattr(model, "hf_device_map")
                    else device
                )
                batch = {k: v.to(tgt_dev) for k, v in batch.items()}
                model(**batch)

        for h in hooks:
            h.remove()

        for name, _ in group_layers:
            pruners[name].ALPS_admm(sparsity, nm_n=(prunen or 0), nm_m=(prunem or 0), rho=rho, max_iter=max_iter)
            pruners[name].free()
            del pruners[name]
            torch.cuda.empty_cache()
            total_done += 1
            print(f"[ALPS] layer {total_done}/{len(layers)} pruned")

        torch.cuda.empty_cache()

    for idx, (lname, layer) in enumerate(layers):
        mem = _estimate_alps_gb(layer) * 1.5  # extra margin for ALPS's per-layer ADMM working set
        if mem > memory_limit_gb:
            print(f"[ALPS] WARNING: single layer '{lname}' requires {mem:.2f} GB > budget {memory_limit_gb} GB; handling it alone.")
            if group:
                _process_group(group, idx - len(group))
                group, group_mem = [], 0.0
            _process_group([(lname, layer)], idx)
            continue

        if group_mem + mem <= memory_limit_gb:
            group.append((lname, layer))
            group_mem += mem
        else:
            _process_group(group, idx - len(group))
            group, group_mem = [(lname, layer)], mem

    _process_group(group, len(layers) - len(group))

    realised = compute_sparsity(model)
    print(f"[ALPS] realised sparsity: {realised*100:.2f}%")