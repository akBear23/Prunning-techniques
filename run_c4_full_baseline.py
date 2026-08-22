#!/usr/bin/env python3
"""
C4 baseline: DeepSeek-R1-Distill-Qwen (1.5B, 7B, 14B) at 40% and 50%.
SparseGPT with C4 calibration, eval MATH500 + LCB + AIME25.
Matches OBC main pruning_results table setup.
"""
import subprocess, os, json, time, glob
from datetime import datetime, timezone

PROJECT = "/mnt/data/lannth/COMP/RAC/RAC/open-r1-main"
os.chdir(PROJECT)

MODELS = {
    "1.5B": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 30),
    "7B":   ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",   60),
    "14B":  ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",  70),
}
SPARSITIES = [0.4, 0.5]

CONDA_RAC = "/home/lannth/miniconda3/envs/rac/bin/python"
TMP = f"TMPDIR={PROJECT}/.tmp_scratch"
TRITON = f"TRITON_CACHE_DIR={PROJECT}/.triton_cache"

EVAL_TASKS = {
    "math500": "lighteval|math_500|0",
    "lcb":     "extended|lcb:codegeneration|0",
    "aime25":  "lighteval|aime25|0",
}

def ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def run(cmd, desc):
    print(f"[{ts()}] {desc}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, executable="/bin/bash")
    elapsed = time.time() - t0
    status = "OK" if r.returncode == 0 else f"FAIL({r.returncode})"
    print(f"[{ts()}]   {status} in {elapsed/60:.1f} min", flush=True)
    return r.returncode


def prune_c4(model_name, sparsity, output_dir, mem_gb=30):
    """Prune with SparseGPT + C4 calibration."""
    if os.path.exists(f"{output_dir}/model.safetensors.index.json"):
        print(f"[{ts()}] [skip] C4 prune -> {output_dir} (exists)")
        return 0
    sp = sparsity
    cmd = (
        f"cd {PROJECT} && {CONDA_RAC} run_c4_baseline.py "
        f"--model_name {model_name} --sparsity {sp} "
        f"--output_dir {output_dir} "
        f"--n_calib_samples 128 --calib_seq_len 2048 "
        f"--device cuda --dtype bfloat16 --memory_limit_gb {mem_gb}"
    )
    return run(cmd, f"C4 prune {output_dir}")


def eval_benchmark(model_path, output_dir, task, label):
    """Run a lighteval benchmark on a pruned model."""
    # Check if already done
    existing = glob.glob(f"{output_dir}/**/results_*.json", recursive=True)
    if existing:
        print(f"[{ts()}] [skip] eval {label} -> {output_dir} (exists)")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    model_args = (
        f"model_name={model_path},dtype=bfloat16,trust_remote_code=true,"
        f"max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=1,"
        f"generation_parameters={{max_new_tokens:32768,temperature:0.6,top_p:0.95}}"
    )
    cmd = (
        f"source /home/lannth/miniconda3/bin/activate lvr && "
        f"cd {PROJECT} && "
        f"VLLM_WORKER_MULTIPROC_METHOD=spawn {TMP} {TRITON} "
        f"lighteval vllm \"{model_args}\" \"{task}\" "
        f"--output-dir \"{output_dir}\" --save-details"
    )
    return run(cmd, f"eval {label}")


def extract_result(eval_dir):
    """Extract acc@1 and stderr from a lighteval result dir."""
    rjs = glob.glob(f"{eval_dir}/**/results_*.json", recursive=True)
    if not rjs:
        return None, None, None
    d = json.load(open(rjs[0]))
    results = d.get("results", {})
    for task_key, task_data in results.items():
        if task_key == "all":
            continue
        if isinstance(task_data, dict):
            # Try common metric names
            acc = (task_data.get("pass@k:k=1&n=1") or
                   task_data.get("codegen_pass@1:16") or
                   task_data.get("acc"))
            se  = (task_data.get("pass@k:k=1&n=1_stderr") or
                   task_data.get("codegen_pass@1:16_stderr") or
                   task_data.get("acc_stderr"))
            rt = float(d.get("config_general", {}).get("total_evaluation_time_secondes", 0)) / 60
            return acc, se, rt
    return None, None, None


# ── Main loop ─────────────────────────────────────────────────
all_results = []

for size_label, (model_name, mem_gb) in MODELS.items():
    for sp in SPARSITIES:
        sp_pct = int(sp * 100)
        size_short = size_label.replace(".", "")
        out_dir = f"{PROJECT}/models/c4_{size_short}_sparse{sp_pct:02d}"

        # 1. Prune
        rc = prune_c4(model_name, sp, out_dir, mem_gb)
        if rc != 0:
            print(f"[{ts()}] SKIPPING evals for C4 {size_label}@{sp_pct}% (prune failed)")
            continue

        # 2. Eval all three benchmarks
        for bench_name, task_str in EVAL_TASKS.items():
            eval_dir = f"{PROJECT}/results/{bench_name}/c4_{size_short}_sparse{sp_pct:02d}"
            eval_benchmark(out_dir, eval_dir, task_str, f"C4 {size_label}@{sp_pct}% {bench_name}")

        # 3. Extract results
        for bench_name in EVAL_TASKS:
            eval_dir = f"{PROJECT}/results/{bench_name}/c4_{size_short}_sparse{sp_pct:02d}"
            acc, se, rt = extract_result(eval_dir)
            all_results.append({
                "size": size_label, "sparsity": sp_pct, "bench": bench_name,
                "acc": acc, "stderr": se, "runtime_min": rt,
            })

# ── Final summary ─────────────────────────────────────────────
print(f"\n[{ts()}] ===== C4 BASELINE SUMMARY =====")
print(f"{'Size':>6s} {'Sp%':>4s} {'Bench':>8s} {'Acc':>10s} {'Stderr':>10s} {'Runtime(min)':>14s}")
print("-" * 65)
for r in sorted(all_results, key=lambda x: (x["size"], x["sparsity"], x["bench"])):
    acc_s = f"{r['acc']:.4f}" if isinstance(r['acc'], (int, float)) else str(r['acc'])
    se_s  = f"{r['stderr']:.4f}" if isinstance(r['stderr'], (int, float)) else str(r['stderr'] or "-")
    rt_s  = f"{r['runtime_min']:.1f}" if isinstance(r['runtime_min'], (int, float)) else "-"
    print(f"{r['size']:>6s} {r['sparsity']:>4d} {r['bench']:>8s} {acc_s:>10s} {se_s:>10s} {rt_s:>14s}")
print(f"[{ts()}] ===== C4 baselines DONE =====")
