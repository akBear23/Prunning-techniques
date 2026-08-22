#!/usr/bin/env python3
"""C4 baseline: DeepSeek-R1-Distill-Qwen-1.5B at 40% and 50% with SparseGPT + C4 calib."""
import subprocess, sys, os, json, time
from datetime import datetime, timezone

PROJECT = "/mnt/data/lannth/COMP/RAC/RAC/open-r1-main"
os.chdir(PROJECT)
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CONDA_RAC = "/home/lannth/miniconda3/envs/rac/bin/python"

def ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def run(cmd, desc):
    print(f"[{ts()}] {desc}", flush=True)
    r = subprocess.run(cmd, shell=True, executable="/bin/bash")
    print(f"[{ts()}]   exit={r.returncode}", flush=True)
    return r.returncode

for sp in [0.4, 0.5]:
    sp_pct = int(sp * 100)
    out = f"{PROJECT}/models/c4_1.5B_sparse{sp_pct:02d}"
    idx = f"{out}/model.safetensors.index.json"

    if os.path.exists(idx):
        print(f"[{ts()}] [skip] C4 prune 1.5B@{sp_pct}% -> {out}")
    else:
        t0 = time.time()
        run(f"{CONDA_RAC} {PROJECT}/run_c4_baseline.py --model_name {MODEL} --sparsity {sp} --output_dir {out} --n_calib_samples 128 --calib_seq_len 2048 --device auto --dtype bfloat16 --memory_limit_gb 30",
            f"C4 prune 1.5B@{sp_pct}%")
        print(f"[{ts()}]   [timing] C4 prune {sp_pct}% took {(time.time()-t0)/60:.1f} min")

    # Eval MATH500
    eval_dir = f"{PROJECT}/results/math500/c4_1.5B_sparse{sp_pct:02d}"
    eval_json = f"{eval_dir}/results/{out}/results_placeholder.json"
    if os.path.exists(eval_dir) and os.listdir(eval_dir):
        print(f"[{ts()}] [skip] eval C4 1.5B@{sp_pct}% -> {eval_dir}")
    else:
        t0 = time.time()
        cmd = (
            f"source /home/lannth/miniconda3/bin/activate lvr && cd {PROJECT} && "
            f"VLLM_WORKER_MULTIPROC_METHOD=spawn TMPDIR={PROJECT}/.tmp_scratch TRITON_CACHE_DIR={PROJECT}/.triton_cache "
            f"lighteval vllm "
            f"\"model_name={out},dtype=bfloat16,trust_remote_code=true,max_model_length=32768,"
            f"gpu_memory_utilization=0.8,data_parallel_size=1,"
            f"generation_parameters={{max_new_tokens:32768,temperature:0.6,top_p:0.95}}\" "
            f"\"lighteval|math_500|0\" --output-dir \"{eval_dir}\" --save-details"
        )
        run(cmd, f"eval C4 1.5B@{sp_pct}%")
        print(f"[{ts()}]   [timing] eval C4 {sp_pct}% took {(time.time()-t0)/60:.1f} min")

    # Report result
    try:
        rjs = list(__import__("glob").glob(f"{eval_dir}/**/results_*.json", recursive=True))
        if rjs:
            d = json.load(open(rjs[0]))
            acc = list(d.get("results",{}).values())[0].get("pass@k:k=1&n=1","?")
            rt = float(d.get("config_general",{}).get("total_evaluation_time_secondes",0))/60
            print(f"[{ts()}] [RESULT] C4 1.5B@{sp_pct}%: acc@1={acc} runtime_min={rt:.1f}", flush=True)
    except Exception as e:
        print(f"[{ts()}] [RESULT] C4 1.5B@{sp_pct}%: ERROR extracting - {e}")

print(f"\n[{ts()}] ===== C4 baselines DONE =====")
