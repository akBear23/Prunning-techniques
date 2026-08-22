#!/usr/bin/env python3
"""
Re-prune Wanda baselines + eval MATH500 on GPU 28682 (zellij 15b).
C4+Wanda and SSGR+Wanda for DeepSeek-R1-Distill-Qwen 1.5B/7B at 40%/50%.
"""
import subprocess, os, json, time, pickle, glob, sys, shutil
from datetime import datetime, timezone

PROJECT = "/mnt/data/lannth/COMP/RAC/RAC/open-r1-main"
OBC_DIR = "/mnt/data/vhoangth2/repos/OBC"
MODELS_DIR = f"{OBC_DIR}/models"
RESULTS_DIR = f"{OBC_DIR}/results/math500"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CONDA_RAC = "/home/lannth/miniconda3/envs/rac/bin/python"
PYTHONPATH_ENV = (f"PYTHONPATH=src:src/open_r1:{PROJECT}/src:{PROJECT}/src/open_r1:"
                  f"{PROJECT}/src/open_r1/open_r1_trl")

MODEL_CFG = {
    "1.5B": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
             f"{PROJECT}/models/wanda_1.5B_sparse04/pairs_annotated.pkl", 30, "15B"),
    "7B":   ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
             f"{PROJECT}/models/daoc_full_7B_sparse04/pairs_annotated.pkl", 35, "7B"),
}
SPARSITIES = [0.4, 0.5]
C4_SCRIPT = f"{OBC_DIR}/run_c4_wanda_alps.py"

def ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def run(cmd, desc, tm=180):
    print(f"[{ts()}] {desc}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, executable="/bin/bash", timeout=tm*60, cwd=PROJECT)
    e = time.time() - t0
    s = "OK" if r.returncode==0 else f"FAIL({r.returncode})"
    print(f"[{ts()}]   {s} in {e/60:.1f} min", flush=True)
    return r.returncode

def ss(sl): return sl.replace(".", "")

def mdir(cond, sl, sp):
    return f"{MODELS_DIR}/{cond}_wanda_{ss(sl)}_sparse{int(sp*100):02d}"

def edir(cond, sl, sp):
    return f"{RESULTS_DIR}/{cond}_wanda_{ss(sl)}_sparse{int(sp*100):02d}"

def build_pairs_cache(daoc_pkl, out_dir):
    cp = f"{out_dir}/pairs_cache.pkl"
    if os.path.exists(cp): return
    sys.path.insert(0, f"{PROJECT}/src"); sys.path.insert(0, f"{PROJECT}/src/open_r1")
    with open(daoc_pkl, "rb") as f: pairs = pickle.load(f)
    print(f"[{ts()}] Loaded {len(pairs)} DAOC pairs")
    os.makedirs(out_dir, exist_ok=True)
    with open(cp, "wb") as f: pickle.dump({"pairs":pairs,"next_pid":len(pairs)}, f)

def prune_c4_wanda(model_name, sl, sp, mem_gb):
    out = mdir("c4", sl, sp)
    # Always re-prune (delete old corrupted model)
    if os.path.exists(f"{out}/model.safetensors"):
        print(f"[{ts()}] Removing corrupted model: {out}")
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    cmd = (
        f"cd {PROJECT} && {PYTHONPATH_ENV} {CONDA_RAC} {C4_SCRIPT} "
        f"--model_name {model_name} --sparsity {sp} --backend wanda "
        f"--output_dir {out} --device cuda --dtype bfloat16 --scope all "
        f"--memory_limit_gb {mem_gb} --n_calib_samples 128 --calib_seq_len 2048"
    )
    return run(cmd, f"C4+Wanda prune {sl}@{int(sp*100)}%")

def prune_ssgr_wanda(model_name, sl, sp, daoc_pkl, mem_gb):
    out = mdir("ssgr", sl, sp)
    if os.path.exists(f"{out}/model.safetensors"):
        print(f"[{ts()}] Removing corrupted model: {out}")
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    build_pairs_cache(daoc_pkl, out)
    cmd = (
        f"cd {PROJECT} && {PYTHONPATH_ENV} {CONDA_RAC} -m open_r1.daoc_prune.run_phase1 "
        f"--model_name {model_name} --dataset open-r1/OpenR1-Math-220k --task_type math "
        f"--sparsity {sp} --n_pairs 32 --n_rollouts 16 "
        f"--temperature 0.8 --max_new_tokens 32768 "
        f"--condition ssgr --ssgr_min_len_pct 25.0 --ssgr_max_len_pct 75.0 "
        f"--difficulty_lo 0.2 --difficulty_hi 0.8 "
        f"--backend wanda --scope all --memory_limit_gb {mem_gb} "
        f"--output_dir {out} --device cuda --dtype bfloat16 "
        f"--no_store_stop_logits --no_save_rollouts "
        f"--pairs_cache {out}/pairs_cache.pkl"
    )
    return run(cmd, f"SSGR+Wanda prune {sl}@{int(sp*100)}%")

def eval_math500(model_path, output_dir, label):
    rjs = glob.glob(f"{output_dir}/**/results_*.json", recursive=True)
    if rjs:
        print(f"[{ts()}] [skip] eval {label}")
        return 0
    os.makedirs(output_dir, exist_ok=True)
    model_args = (
        f"model_name={model_path},dtype=bfloat16,trust_remote_code=true,"
        f"max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=1,"
        f"generation_parameters={{max_new_tokens:32768,temperature:0.6,top_p:0.95}}"
    )
    cmd = (
        f"source /home/lannth/miniconda3/bin/activate lvr && cd {PROJECT} && "
        f"VLLM_WORKER_MULTIPROC_METHOD=spawn "
        f"TMPDIR={PROJECT}/.tmp_scratch TRITON_CACHE_DIR={PROJECT}/.triton_cache "
        f"lighteval vllm \"{model_args}\" \"lighteval|math_500|0\" "
        f"--output-dir \"{output_dir}\" --save-details"
    )
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
    for sl, (model_name, daoc_pkl, mem_gb, _) in MODEL_CFG.items():
        for sp in SPARSITIES:
            sp_pct = int(sp * 100)
            print(f"\n{'='*60}")
            print(f"[{ts()}] {cond}+Wanda | {sl} | {sp_pct}%")
            print(f"{'='*60}")

            # Prune
            if cond == "c4":
                rc = prune_c4_wanda(model_name, sl, sp, mem_gb)
            else:
                rc = prune_ssgr_wanda(model_name, sl, sp, daoc_pkl, mem_gb)
            if rc != 0:
                print(f"[{ts()}] SKIP eval")
                continue

            # Eval
            out = mdir(cond, sl, sp)
            ed = edir(cond, sl, sp)
            eval_math500(out, ed, f"{cond}+Wanda {sl}@{sp_pct}%")

            acc, se, rt = extract(ed)
            all_results.append({"cond":cond,"size":sl,"sparsity":sp_pct,"acc":acc,"stderr":se,"rt":rt})
            if acc is not None:
                print(f"[{ts()}] [RESULT] {cond}+Wanda {sl}@{sp_pct}%: acc={acc:.4f}")
            else:
                print(f"[{ts()}] [RESULT] {cond}+Wanda {sl}@{sp_pct}%: NO RESULT (eval may have failed)")

print(f"\n{'='*70}")
print(f"[{ts()}] ===== WANDA BASELINES SUMMARY =====")
print(f"{'Calib':>6s} {'Size':>6s} {'Sp%':>4s} {'Acc':>10s} {'Stderr':>10s} {'Runtime':>14s}")
print("-"*70)
for r in sorted(all_results, key=lambda x: (x["cond"],x["size"],x["sparsity"])):
    acc_s = f"{r['acc']:.4f}" if isinstance(r['acc'],(int,float)) else str(r['acc'])
    se_s = f"{r['stderr']:.4f}" if isinstance(r['stderr'],(int,float)) else str(r['stderr'] or "-")
    rt_s = f"{r['rt']:.1f}" if isinstance(r['rt'],(int,float)) else "-"
    print(f"{r['cond']:>6s} {r['size']:>6s} {r['sparsity']:>4d} {acc_s:>10s} {se_s:>10s} {rt_s:>14s}")
print(f"[{ts()}] ===== DONE =====")
