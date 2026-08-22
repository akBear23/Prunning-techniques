#!/usr/bin/env python3
"""C4 14B retry — only 14B@40% and 14B@50% (GPU 28681)."""
import subprocess, os, json, time, glob
from datetime import datetime, timezone

PROJECT = "/mnt/data/lannth/COMP/RAC/RAC/open-r1-main"
os.chdir(PROJECT)

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
MEM_GB = 40  # Lowered from 70 to force layer grouping (avoids OOM on 79GB GPU)
SPARSITIES = [0.4, 0.5]
SIZE_LABEL = "14B"
SIZE_SHORT = "14B"

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

def model_exists(out_dir):
    """Check for either safetensors or index.json."""
    return (os.path.exists(f"{out_dir}/model.safetensors.index.json") or
            os.path.exists(f"{out_dir}/model.safetensors"))

def prune_c4(sparsity, output_dir):
    if model_exists(output_dir):
        print(f"[{ts()}] [skip] C4 prune -> {output_dir} (exists)")
        return 0
    cmd = (
        f"cd {PROJECT} && {CONDA_RAC} run_c4_baseline.py "
        f"--model_name {MODEL_NAME} --sparsity {sparsity} "
        f"--output_dir {output_dir} "
        f"--n_calib_samples 128 --calib_seq_len 2048 "
        f"--device cuda --dtype bfloat16 --memory_limit_gb {MEM_GB}"
    )
    return run(cmd, f"C4 prune {SIZE_LABEL}@{int(sparsity*100)}%")

def eval_benchmark(model_path, output_dir, task_str, label):
    existing = glob.glob(f"{output_dir}/**/results_*.json", recursive=True)
    if existing:
        print(f"[{ts()}] [skip] eval {label} (exists)")
        return 0
    os.makedirs(output_dir, exist_ok=True)
    model_args = (
        f"model_name={model_path},dtype=bfloat16,trust_remote_code=true,"
        f"max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=1,"
        f"generation_parameters={{max_new_tokens:32768,temperature:0.6,top_p:0.95}}"
    )
    cmd = (
        f"source /home/lannth/miniconda3/bin/activate lvr && cd {PROJECT} && "
        f"VLLM_WORKER_MULTIPROC_METHOD=spawn {TMP} {TRITON} "
        f"lighteval vllm \"{model_args}\" \"{task_str}\" "
        f"--output-dir \"{output_dir}\" --save-details"
    )
    return run(cmd, label)

def extract_result(eval_dir):
    rjs = glob.glob(f"{eval_dir}/**/results_*.json", recursive=True)
    if not rjs:
        return None, None, None
    d = json.load(open(rjs[0]))
    results = d.get("results", {})
    for task_key, task_data in results.items():
        if task_key == "all":
            continue
        if isinstance(task_data, dict):
            acc = (task_data.get("pass@k:k=1&n=1") or
                   task_data.get("codegen_pass@1:16") or
                   task_data.get("acc"))
            se  = (task_data.get("pass@k:k=1&n=1_stderr") or
                   task_data.get("codegen_pass@1:16_stderr") or
                   task_data.get("acc_stderr"))
            rt = float(d.get("config_general", {}).get("total_evaluation_time_secondes", 0)) / 60
            return acc, se, rt
    return None, None, None

# ── Main ───────────────────────────────────────────────────
all_results = []

for sp in SPARSITIES:
    sp_pct = int(sp * 100)
    out_dir = f"{PROJECT}/models/c4_{SIZE_SHORT}_sparse{sp_pct:02d}"

    # 1. Prune
    rc = prune_c4(sp, out_dir)
    if rc != 0:
        print(f"[{ts()}] SKIPPING evals for C4 {SIZE_LABEL}@{sp_pct}% (prune failed)")
        continue

    # 2. Eval all three benchmarks
    for bench_name, task_str in EVAL_TASKS.items():
        eval_dir = f"{PROJECT}/results/{bench_name}/c4_{SIZE_SHORT}_sparse{sp_pct:02d}"
        eval_benchmark(out_dir, eval_dir, task_str, f"C4 {SIZE_LABEL}@{sp_pct}% {bench_name}")

    # 3. Extract results
    for bench_name in EVAL_TASKS:
        eval_dir = f"{PROJECT}/results/{bench_name}/c4_{SIZE_SHORT}_sparse{sp_pct:02d}"
        acc, se, rt = extract_result(eval_dir)
        all_results.append({
            "size": SIZE_LABEL, "sparsity": sp_pct, "bench": bench_name,
            "acc": acc, "stderr": se, "runtime_min": rt,
        })

    # Report this cell
    print(f"\n[{ts()}] C4 {SIZE_LABEL}@{sp_pct}% RESULTS:")
    for r in all_results[-3:]:
        acc_s = f"{r['acc']:.4f}" if isinstance(r['acc'], (int, float)) else str(r['acc'])
        print(f"  {r['bench']:>8s}: acc={acc_s}")

print(f"\n[{ts()}] ===== C4 14B retry DONE =====")
for r in sorted(all_results, key=lambda x: (x["sparsity"], x["bench"])):
    acc_s = f"{r['acc']:.4f}" if isinstance(r['acc'], (int, float)) else str(r['acc'])
    se_s  = f"{r['stderr']:.4f}" if isinstance(r['stderr'], (int, float)) else str(r['stderr'] or "-")
    rt_s  = f"{r['runtime_min']:.1f}" if isinstance(r['runtime_min'], (int, float)) else "-"
    print(f"C4 {r['size']:>4s} {r['sparsity']:>4d}% {r['bench']:>8s}: acc={acc_s}, stderr={se_s}, runtime={rt_s}m")
