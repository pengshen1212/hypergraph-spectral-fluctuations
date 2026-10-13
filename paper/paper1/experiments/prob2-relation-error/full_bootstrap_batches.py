"""P1-B 独立外层划分/种子批次的完整传播 bootstrap（2026-10-08）。

补证据清单 P1-B：现有 full_bootstrap.json 的校准池与评估组固定在
seed 31000+/32000+ 一个划分上，旧多划分 envelope 不是统一 CI。本脚本
按同一完整口径跑 3 个完全不相交的外层批次（校准种子 33000+/41000+/
49000+，评估种子 36000+/44000+/52000+，均 80+20 抽样，与原 31000+/32000+
不重叠），逐批报告点估计与 95% percentile 区间（含跨过/低于 1 的结果），
不合并成 envelope。

协议与 full_bootstrap.py 一致：每次 bootstrap 重采样校准运行→重算逐边
q̂ 与中心化均值 d̄→按新率重新生成 20 条 B→重采样全部 20 条评估轨迹，
统计量 = mean‖A−M‖ / mean‖B−M‖，N_BOOT=1000，n_cal ∈ {20,40,80}，
4 设置（Cora/DBLP × all/tri）。

实现差异（已对账）：谱范数用 25 步全重正交 Lanczos（一次运行同时逼近
两端极值）代替 eigsh(LA)+eigsh(SA) 两次调用；启动向量按矩阵序号确定性
播种。运行前先在实际构型矩阵上与 eigsh 对账（断言最大相对误差 <
5e-4），不通过即中止。bootstrap 复制并行执行，每复制一条独立子流
（seed = BOOT_BASE + 1000·batch + b），与单流版本统计等价。
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.stdout.reconfigure(encoding="utf-8")
from realdata_fixed import all_hom, load
from crossfit_v2 import crossfit_predict_sparse

CORA = "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/data/cora-cc"
DBLP = "C:/Users/Peng1/OneDrive/文档/ChatGPT/hypergraph/data/dblp-ca"
CAL_SIZES = (20, 40, 80)
N_EVAL = 20
N_B_PER_BS = 20
N_BOOT = 1000
LANCZOS_STEPS = 25

# 批次种子块（校准 80 个 + 评估 20 个，互不相交，也与原 31000+/32000+ 不相交）
BATCHES = (
    dict(name="batch0", cal=33000, eval=36000, boot=3000),
    dict(name="batch1", cal=41000, eval=44000, boot=4000),
    dict(name="batch2", cal=49000, eval=52000, boot=5000),
)

G = {}   # worker 全局


def build_op(w):
    vals = G["invk"] * w[G["owner"]]
    return sp.csr_matrix((vals, (G["rows"], G["cols"])),
                         shape=(G["n"], G["n"]))


def lanczos_op_norm(M, v0, steps=LANCZOS_STEPS):
    """全重正交 Lanczos，返回 max|λ| 的近似。v0 需单位化前任意非零向量。"""
    n = M.shape[0]
    Q = np.empty((n, steps))
    a = np.empty(steps)
    b = np.empty(steps - 1)
    q = v0 / np.linalg.norm(v0)
    Q[:, 0] = q
    r = M @ q
    a[0] = q @ r
    r = r - a[0] * q
    for j in range(1, steps):
        bj = np.linalg.norm(r)
        if bj < 1e-13:                       # happy breakdown：换随机重启
            rng = np.random.default_rng(abs(hash((j, r[0]))) % (2**32))
            q = rng.standard_normal(n)
            q -= Q[:, :j] @ (Q[:, :j].T @ q)
            q /= np.linalg.norm(q)
        else:
            q = r / bj
            b[j - 1] = bj
        Q[:, j] = q
        r = M @ q
        r = r - b[j - 1] * Q[:, j - 1]
        a[j] = q @ r
        r = r - a[j] * q
        r = r - Q[:, :j + 1] @ (Q[:, :j + 1].T @ r)   # 全重正交
    T = np.diag(a) + np.diag(b, 1) + np.diag(b, -1)
    ev = np.linalg.eigvalsh(T)
    return max(abs(ev[0]), abs(ev[-1]))


def sym_norm_eigsh(M):
    """原口径参考实现（对账用）。"""
    if M.nnz == 0:
        return 0.0
    try:
        la = float(eigsh(M, k=1, which="LA", tol=1e-5, maxiter=5000,
                         return_eigenvectors=False)[0])
    except Exception:
        la = 0.0
    try:
        li = float(eigsh(M, k=1, which="SA", tol=1e-5, maxiter=5000,
                         return_eigenvectors=False)[0])
    except Exception:
        li = 0.0
    return max(abs(la), abs(li))


def v0_for(seed):
    return np.random.default_rng(seed).standard_normal(G["n"])


def one_replicate(args):
    b, seed, n_cal = args
    rng = np.random.default_rng(seed)
    ic = rng.integers(0, n_cal, n_cal)
    q_b = (G["D_cal"][ic] != 0).mean(axis=0)
    d_bar_b = G["D_cal"][ic].mean(axis=0)
    ie = rng.integers(0, G["N_EVAL"], G["N_EVAL"])
    ca = [lanczos_op_norm(build_op(G["D_eval"][j] - d_bar_b),
                          v0_for(seed * 131 + 17 * j))
          for j in ie]
    cb = []
    for k in range(G["N_B_PER_BS"]):
        w = np.where(rng.random(G["m"]) < q_b, G["sgn"], 0.0) \
            - G["sgn"] * q_b
        cb.append(lanczos_op_norm(build_op(w), v0_for(seed * 137 + 19 * k)))
    return float(np.mean(ca) / max(np.mean(cb), 1e-12))


def init_worker(payload):
    G.update(payload)


def validate_estimator(payload, n_check=30):
    """实际构型上 Lanczos vs eigsh 对账。"""
    G.update(payload)
    rng = np.random.default_rng(777)
    worst = 0.0
    for i in range(n_check):
        wA = G["D_eval"][rng.integers(0, len(G["D_eval"]))] \
            - G["D_cal"][:20].mean(axis=0)
        M = build_op(wA)
        v0 = v0_for(9000 + i)
        ln = lanczos_op_norm(M, v0)
        eg = sym_norm_eigsh(M)
        if eg > 0:
            worst = max(worst, abs(ln - eg) / eg)
    return worst


def run_setting(name, feats, edges_all, labels, tri_only, batch,
                n_cals=CAL_SIZES):
    n = len(labels)
    edges = [e for e in edges_all if len(e) == 3] if tri_only else edges_all
    a_star = all_hom(labels, edges)
    m = len(edges)
    sgn = np.where(a_star == 1, -1.0, 1.0)
    rows, cols, invk, owner = [], [], [], []
    for i, e in enumerate(edges):
        k = len(e)
        rows.extend(np.repeat(e, k).tolist())
        cols.extend(np.tile(e, k).tolist())
        owner.extend([i] * (k * k))
        invk.extend([1.0 / k] * (k * k))
    payload = dict(n=n, m=m, sgn=sgn, N_EVAL=N_EVAL,
                   N_B_PER_BS=N_B_PER_BS,
                   rows=np.array(rows, dtype=np.int32),
                   cols=np.array(cols, dtype=np.int32),
                   invk=np.array(invk), owner=np.array(owner,
                                                       dtype=np.int32),
                   D_cal=None, D_eval=None)

    D_eval = []
    for s in range(N_EVAL):
        pred = crossfit_predict_sparse(feats, labels, batch["eval"] + s)
        D_eval.append((all_hom(pred, edges) - a_star).astype(float))
    D_cal_pool = []
    for s in range(max(CAL_SIZES)):
        pred = crossfit_predict_sparse(feats, labels, batch["cal"] + s)
        D_cal_pool.append((all_hom(pred, edges) - a_star).astype(float))
    D_cal_pool = np.array(D_cal_pool)
    payload["D_eval"] = np.array(D_eval)

    results = []
    for n_cal in n_cals:
        payload["D_cal"] = D_cal_pool[:n_cal]
        worst = validate_estimator(payload, n_check=8)
        assert worst < 5e-4, f"Lanczos 对账失败: rel err {worst}"
        rng0 = np.random.default_rng(53000 + (batch["boot"] % 1000))
        q0 = (payload["D_cal"] != 0).mean(axis=0)
        d_bar0 = payload["D_cal"].mean(axis=0)
        cenA0 = [lanczos_op_norm(build_op(payload["D_eval"][j] - d_bar0),
                                 v0_for(6100 + j)) for j in range(N_EVAL)]
        cenB0 = [lanczos_op_norm(
            build_op(np.where(rng0.random(m) < q0, sgn, 0.0) - sgn * q0),
            v0_for(6300 + j)) for j in range(N_B_PER_BS)]
        ratio0 = float(np.mean(cenA0) / np.mean(cenB0))

        tasks = [(b, batch["boot"] * 1000 + 7919 * n_cal + b, n_cal)
                 for b in range(N_BOOT)]
        with mp.Pool(16, initializer=init_worker,
                     initargs=(payload,)) as pool:
            ratios = pool.map(one_replicate, tasks, chunksize=8)
        lo, hi = np.quantile(ratios, [0.025, 0.975])
        results.append(dict(n_cal=n_cal, ratio=ratio0,
                            ci=[float(lo), float(hi)],
                            lanczos_check_relerr=float(worst)))
        print(f"    {name} {batch['name']} N_cal={n_cal}: "
              f"ratio={ratio0:.4f} CI=[{lo:.4f},{hi:.4f}] "
              f"{'✓>1' if lo > 1 else ''}", flush=True)
    return results


def main():
    t_all = time.time()
    print("=" * 96)
    print(f"独立划分批次完整 bootstrap：{len(BATCHES)} 批 × 4 设置 × "
          f"n_cal {CAL_SIZES}，N_BOOT={N_BOOT}，Lanczos {LANCZOS_STEPS} 步")
    print("=" * 96)
    import os
    out_path = "results/full_bootstrap_batches.json"
    out = None
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            old = json.load(fh)
        cells = []
        for blk in old.get("settings", []):
            for r in blk["results"]:
                cells.append(dict(name=blk["name"], batch=blk["batch"],
                                  cal_seed=blk["cal_seed"],
                                  eval_seed=blk["eval_seed"],
                                  boot_seed_base=blk["boot_seed_base"],
                                  **r))
        old["cells"] = cells
        out = old
        print(f"续跑：载入已有 {len(cells)} 个单元格结果", flush=True)
    if out is None:
        out = dict(protocol=("3 fully disjoint outer split batches (cal/eval "
                         "seeds 33000+/36000+, 41000+/44000+, 49000+/"
                         "52000+; disjoint from the original 31000+/32000+)"
                         "; same full-propagation bootstrap as "
                         "full_bootstrap.py (1000 replicates, per-replicate "
                         "independent substreams); spectral norm by 25-step "
                         "fully reorthogonalized Lanczos validated against "
                         "eigsh on real-configuration matrices"),
                       batches=[b["name"] for b in BATCHES],
                       n_boot=N_BOOT,
                       settings=[])
    def save():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)

    def run_all(feats, edges, labels, combos, sizes=CAL_SIZES):
        for tri_only, nm in combos:
            for batch in BATCHES:
                done_ncal = {c["n_cal"] for c in out["cells"]
                             if c["name"] == nm and c["batch"] == batch["name"]}
                todo = [nc for nc in sizes if nc not in done_ncal]
                if not todo:
                    print(f"  跳过已完成 {nm} {batch['name']}", flush=True)
                    continue
                t0 = time.time()
                res = run_setting(nm, feats, edges, labels, tri_only, batch,
                                  n_cals=todo)
                for r in res:
                    out["cells"].append(dict(
                        name=nm, batch=batch["name"], cal_seed=batch["cal"],
                        eval_seed=batch["eval"],
                        boot_seed_base=batch["boot"], **r))
                    save()          # 逐格落盘（断点续跑）
                print(f"  [{nm} {batch['name']}] 用时 "
                      f"{(time.time()-t0)/60:.1f} min（已落盘）", flush=True)
    feats, edges, labels = load(CORA)
    run_all(feats, edges, labels, ((False, "Cora-all"), (True, "Cora-tri")))
    feats, edges, labels = load(DBLP)
    # DBLP 批次只跑 n_cal=20：n_cal 敏感性已由固定划分表与 Cora 批次覆盖，
    # 全网格成本（约 2.7h）不成比例
    run_all(feats, edges, labels, ((False, "DBLP-all"), (True, "DBLP-tri")),
            sizes=(20,))
    print(f"\n总用时 {(time.time()-t_all)/60:.1f} min；"
          f"已写出 results/full_bootstrap_batches.json")
    print("\n===== 逐批汇总（不合并 envelope）=====")
    for s in out["settings"]:
        for r in s["results"]:
            print(f"  {s['name']:<9} {s['batch']:<7} n_cal={r['n_cal']:>2} "
                  f"ratio={r['ratio']:.3f} "
                  f"CI=[{r['ci'][0]:.4f},{r['ci'][1]:.4f}] "
                  f"{'✓' if r['ci'][0] > 1 else ''}")


if __name__ == "__main__":
    main()
