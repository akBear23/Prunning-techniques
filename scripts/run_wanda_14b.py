#!/usr/bin/env python3
"""Wanda 14B baselines: C4+Wanda and SSGR+Wanda on GPU 28682. MATH500 only."""
import subprocess, os, json, time, pickle, glob, sys
from datetime import datetime, timezone

PROJECT = "/mnt/data/lannth/COMP/RAC/RAC/open-r1-main"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DAOC_PAIRS = f"{PROJECT}/models/wanda_14B_sparse04/pairs_annotated.pkl"
MEM_GB = 60

CONDA_RAC = "/home/lannth/miniconda3/envs/rac/bin/python"
PYTHONPATH_ENV = (f"PYTHONPATH=src:src/open_r1:{PROJECT}/src:{PROJECT}/src/open_r1:"
                  f"{PROJECT}/src/open_r1/open_r1_trl")

os.chdir(PROJECT)

def ts(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def run(cmd, desc, tm=180):
    print(f"[{ts()}] {desc}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, executable="/bin/bash", timeout=tm*60, cwd=PROJECT)
    e = time.time() - t0
    s = "OK" if r.returncode==0 else f"FAIL({r.returncode})"
    print(f"[{ts()}]   {s} in {e/60:.1f} min", flush=True)
    return r.returncode

def build_pairs_cache(daoc_pkl, out_dir):
    cp = f"{out_dir}/pairs_cache.pkl"
    if os.path.exists(cp): return
    sys.path.insert(0, f"{PROJECT}/src"); sys.path.insert(0, f"{PROJECT}/src/open_r1")
    with open(daoc_pkl, "rb") as f: pairs = pickle.load(f)
    print(f"[{ts()}] Loaded {len(pairs)} DAOC pairs")
    os.makedirs(out_dir, exist_ok=True)
    with open(cp, "wb") as f: pickle.dump({"pairs":pairs,"next_pid":len(pairs)}, f)

def prune_c4_wanda(sp):
    out = f"{PROJECT}/models/c4_wanda_14B_sparse{int(sp*100):02d}"
    if os.path.exists(f"{out}/model.safetensors"):
        print(f"[{ts()}] [skip] C4+Wanda prune 14B@{int(sp*100)}%")
        return out, 0
    os.makedirs(out, exist_ok=True)
    cmd = (f"{PYTHONPATH_ENV} {CONDA_RAC} /mnt/data/vhoangth2/repos/OBC/run_c4_wanda_alps.py "
           f"--model_name {MODEL_NAME} --sparsity {sp} --backend wanda "
           f"--output_dir {out} --device cuda --dtype bfloat16 --scope all "
           f"--memory_limit_gb {MEM_GB} --n_calib_samples 128 --calib_seq_len 2048")
    return out, run(cmd, f"C4+Wanda prune 14B@{int(sp*100)}%")

def prune_ssgr_wanda(sp):
    out = f"{PROJECT}/models/ssgr_wanda_14B_sparse{int(sp*100):02d}"
    if os.path.exists(f"{out}/model.safetensors"):
        print(f"[{ts()}] [skip] SSGR+Wanda prune 14B@{int(sp*100)}%")
        return out, 0
    os.makedirs(out, exist_ok=True)
    build_pairs_cache(DAOC_PAIRS, out)
    cmd = (f"{PYTHONPATH_ENV} {CONDA_RAC} -m open_r1.daoc_prune.run_phase1 "
           f"--model_name {MODEL_NAME} --dataset open-r1/OpenR1-Math-220k --task_type math "
           f"--sparsity {sp} --n_pairs 32 --n_rollouts 16 "
           f"--temperature 0.8 --max_new_tokens 32768 "
           f"--condition ssgr --ssgr_min_len_pct 25.0 --ssgr_max_len_pct 75.0 "
           f"--difficulty_lo 0.2 --difficulty_hi 0.8 "
           f"--backend wanda --scope all --memory_limit_gb {MEM_GB} "
           f"--output_dir {out} --device cuda --dtype bfloat16 "
           f"--no_store_stop_logits --no_save_rollouts "
           f"--pairs_cache {out}/pairs_cache.pkl")
    return out, run(cmd, f"SSGR+Wanda prune 14B@{int(sp*100)}%")

def eval_math500(model_path, output_dir, label):
    rjs = glob.glob(f"{output_dir}/**/results_*.json", recursive=True)
    if rjs:
        print(f"[{ts()}] [skip] eval {label}")
        return 0
    os.makedirs(output_dir, exist_ok=True)
    model_args = (f"model_name={model_path},dtype=bfloat16,trust_remote_code=true,"
                  f"max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=1,"
                  f"generation_parameters={{max_new_tokens:32768,temperature:0.6,top_p:0.95}}")
    cmd = (f"source /home/lannth/miniconda3/bin/activate lvr && cd {PROJECT} && "
           f"VLLM_WORKER_MULTIPROC_METHOD=spawn "
           f"TMPDIR={PROJECT}/.tmp_scratch TRITON_CACHE_DIR={PROJECT}/.triton_cache "
           f"lighteval vllm \"{model_args}\" \"lighteval|math_500|0\" "
           f"--output-dir \"{output_dir}\" --save-details")
    return run(cmd, f"eval {label}", tm=120)

def extract(ed):
    rjs = glob.glob(f"{ed}/**/results_*.json", recursive=True)
    if not rjs: return None, None, None
    d = json.load(open(rjs[0]))
    for tk, td in d.get("results",{}).items():
        if tk=="all": continue
        if isinstance(td, dict):
            a = td.get("pass@k:k=1&n=1"); s = td.get("pass@k:k=1&n=1_stderr")
            rt = float(d.get("config_general",{}).get("total_evaluation_time_secondes",0))/60
            return a, s, rt
    return None, None, None

# ═══════════════════════════════════════════════
all_results = []

for cond in ["c4", "ssgr"]:
    for sp in [0.4, 0.5]:
        sp_pct = int(sp*100)
        print(f"\n{'='*60}")
        print(f"[{ts()}] {cond}+Wanda 14B@{sp_pct}%")
        print(f"{'='*60}")

        if cond == "c4":
            out, rc = prune_c4_wanda(sp)
        else:
            out, rc = prune_ssgr_wanda(sp)
        if rc != 0:
            print(f"[{ts()}] SKIP eval — prune failed")
            continue

        ed = f"{PROJECT}/results/math500/{cond}_wanda_14B_sparse{sp_pct:02d}"
        eval_math500(out, ed, f"{cond}+Wanda 14B@{sp_pct}%")

        acc, se, rt = extract(ed)
        all_results.append({"cond":cond,"size":"14B","sparsity":sp_pct,"acc":acc,"stderr":se,"rt":rt})
        if acc is not None:
            print(f"[{ts()}] [RESULT] {cond}+Wanda 14B@{sp_pct}%: acc={acc:.4f} ± {se:.4f} in {rt:.1f}m")
        else:
            print(f"[{ts()}] [RESULT] {cond}+Wanda 14B@{sp_pct}%: NO RESULT")

print(f"\n{'='*70}")
print(f"[{ts()}] Wanda 14B RESULTS:")
for r in sorted(all_results, key=lambda x: (x["cond"],x["sparsity"])):
    print(f"  {r['cond']}+Wanda 14B@{r['sparsity']}%: acc={r['acc']}")
print(f"[{ts()}] DONE")
