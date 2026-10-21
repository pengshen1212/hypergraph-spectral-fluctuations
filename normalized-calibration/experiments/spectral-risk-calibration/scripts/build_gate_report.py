"""Build docs/gate_ab_report.md from S1-S5 results (plan section 12).

Gate A (worth continuing after normalization):
  A1: edge-iid threshold at alpha=5% shows clear over-exceedance (>15%);
  A2: node-calibrated exceedance compatible with target (CP interval covers
      alpha or deviation small);
  A3: mean shift does NOT explain the structural differences
      (shift_scale not >> 1 on the structures where A2 holds);
  A4: influence metric (v_u) tracks structure beyond degree
      (familyA vs familyB_cancel hub contrast + spearman on native).

Gate B (method value):
  B1: normal-mechanism exceedance matches target (from A2);
  B2: detection power at alpha=5% not collapsed (power > zero-alarm floor);
  B3: comparison table vs edge-iid strong baseline (and run-level marked
      pending A3/L3).

Everything below is generated from saved result files; re-running is cheap.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from audit_evidence import summarize_confirmation

HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    lines = ["# 第一篇证据报告：归一化影响、分层校准与运行失配诊断", "",
             "生成脚本：`scripts/build_gate_report.py`；数据：`results/*.csv|json`。",
             "内部投入标准（15% 线为筛选线，非理论界、非发表标准）。",
             "",
             "**单一主线**：节点诱导依赖如何改变归一化谱风险，以及校准需要保留哪一层随机律。",
             "主证据为 S1–S4、S6–S8、S11 与 S12-C1；S9/S10 和 S12-C2/C3 仅作辅助诊断。",
             "本报告读取历史CSV；方向生成与矩阵剩余项的代码修复尚未全量重跑，未来v2结果另存且不与旧表混用。",
             "",
             "**中心分工**：S2 以各臂自身均值中心化（||S−ES_arm||）；",
             "S9a/S10a 是共同确定性基点 S(p) 的扰动响应，并非各臂中心化波动；",
             "监测/校准 S3/S6/S7/S8/S9b/S9c/S10b/S11/S12 使用各单元冻结的 M_hat。",
             "**保证分工**：秩保证条件于独立参考阶段，对校准集与新观测联合取概率；",
             "固定一次阈值的条件超越率可以高于或低于 α。CP 区间只描述该条件率；",
             "区间覆盖 α 不证明等效，点估计低于 α 不等于校准失败。", ""]

    s1 = json.load(open(RESULTS / "exact_checks.json"))
    lines.append("## S1 代数与数值验证")
    lines.append("")
    lines.append(f"- all_pass = **{s1['all_pass']}**")
    c3 = s1["check3_influence_decomposition"]
    lines.append(f"- 归一化一阶分解在小图上精确成立：max|E[R|Y_u=z]| = "
                 f"{c3['E[R|Y_u=z]_maxabs']:.1e}；能量恒等式相对缺口 "
                 f"{c3['energy_identity_gap_frac']:.1e}")
    c4 = s1["check4_mean_separation"]
    lines.append(f"- 固定度两臂均值精确相等（gap={c4['fixed_degree_mean_gap']:.1e}）；"
                 f"动态度均值移位非零 ||E S_A − E S_B||₂ = "
                 f"{c4['dynamic_mean_shift_spectral']:.3f}")
    lines.append("")

    pilot = read_csv(RESULTS / "normalization_pilot.csv")
    lines.append("## S2 三算子对照（B / S0 / S）")
    lines.append("")
    lines.append("| 结构 | 误差律 | ratio_raw | ratio_S0 | ratio_S | 均值移位 ||ES_A−ES_B|| | shift_scale |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in pilot:
        lines.append(
            f"| {r['structure']} | {r['profile']} | "
            f"{fnum(r['ratio_raw']):.2f} | {fnum(r['ratio_S0']):.2f} | "
            f"{fnum(r['ratio_S']):.2f} | {fnum(r['mean_shift_dyn']):.4f} | "
            f"{fnum(r['shift_scale']):.3f} |")
    lines.append("")

    cal_p = RESULTS / "calibration_baselines.csv"
    if cal_p.exists():
        cal = read_csv(cal_p)
    else:
        cal = []
        lines.append("## S3 校准（尚缺：calibration_baselines.csv 生成中）")
        lines.append("")
    normal = [r for r in cal if r["kind"] == "normal"]
    det = [r for r in cal if r["kind"] == "detection"]
    s3_available = bool(normal)

    agg = defaultdict(list)
    for r in normal:
        if fnum(r["alpha"]) == 0.05:
            agg[(r["structure"], r["profile"])].append(r)

    lines.append("## S3 模型内校准（α=5%，批次均值 + 逐批区间端点包络）")
    lines.append("")
    lines.append("方括号为各批 CP 区间的最小下端点和最大上端点，不是合并样本的 95% CI；PASS 是历史内部筛选结果，不是显著性或等效性检验。")
    lines.append("")
    lines.append("| 单元 | node 校准 | edge-iid 校准 | 判定 |")
    lines.append("|---|---|---|---|")
    gate_a1 = gate_a2 = None
    a1_flags, a2_flags = [], []
    cells = {}
    for key, rows in sorted(agg.items()):
        def summ(cname):
            rs = [r for r in rows if r["calibrator"] == cname]
            if not rs:
                return None
            ex = [fnum(r["exceed"]) for r in rs]
            lo = [fnum(r["cp_lo"]) for r in rs]
            hi = [fnum(r["cp_hi"]) for r in rs]
            return sum(ex) / len(ex), min(lo), max(hi)
        n_, e_ = summ("node"), summ("edge-iid")
        judge = ""
        if n_ and e_:
            over = e_[0] > 0.15
            ok = n_[1] <= 0.05 <= n_[2] or abs(n_[0] - 0.05) <= 0.03
            judge = ("PASS" if over and ok else
                     "PARTIAL" if (over or ok) else "CHECK")
            if key[0] in ("cora-cc", "dblp-ca"):
                a1_flags.append(over)
                a2_flags.append(ok)
            cells[key] = (n_, e_, judge)
        ns = f"{n_[0]:.3f} [{n_[1]:.3f},{n_[2]:.3f}]" if n_ else "–"
        es = f"{e_[0]:.3f} [{e_[1]:.3f},{e_[2]:.3f}]" if e_ else "–"
        lines.append(f"| {key[0]}/{key[1]} | {ns} | {es} | {judge} |")
    if a1_flags:
        gate_a1 = all(a1_flags)
        gate_a2 = all(a2_flags)
    lines.append("")
    if s3_available:
        lines.append(f"- **Gate A1**（独立边在 α=5% 明显过度超越 >15%）：**"
                     f"{'达成' if gate_a1 else '未达成'}**")
        lines.append(f"- **Gate A2**（已测原生图满足内部相容筛选线）：**"
                     f"{'达成' if gate_a2 else '未达成'}**")
    else:
        lines.append("- Gate A1/A2：待 S3 数据。")

    # alpha sensitivity table
    lines.append("### α 敏感性（均值超越率）")
    lines.append("")
    lines.append("| 单元 | α | node | edge-iid |")
    lines.append("|---|---|---:|---:|")
    agg2 = defaultdict(lambda: defaultdict(list))
    for r in normal:
        agg2[(r["structure"], r["profile"], fnum(r["alpha"]))][
            r["calibrator"]].append(fnum(r["exceed"]))
    for key, byc in sorted(agg2.items(), key=lambda x: (x[0][0], x[0][1],
                                                        x[0][2])):
        if key[0] in ("familyA_coherent",):
            continue
        n_line = sum(byc.get("node", [])) / max(len(byc.get("node", [])), 1)
        e_line = sum(byc.get("edge-iid", [])) / max(len(byc.get("edge-iid",
                                                                 [])), 1)
        lines.append(f"| {key[0]}/{key[1]} | {key[2]:.2f} | {n_line:.3f} | "
                     f"{e_line:.3f} |")
    lines.append("")
    fa_rows = [r for r in normal if r["structure"] == "familyA_coherent"]
    if fa_rows:
        fa5 = [fnum(r["exceed"]) for r in fa_rows
               if fnum(r["alpha"]) == 0.05 and r["calibrator"] == "node"]
        fa10 = [fnum(r["exceed"]) for r in fa_rows
                if fnum(r["alpha"]) == 0.10 and r["calibrator"] == "node"]
        if fa5 and fa10:
            lines.append("- **familyA 原子性说明（非 bug，结构性质）**：相干族的"
                         "分数尾部由 hub 翻转离散状态主导，最大值处存在约 10% "
                         "质量的原子；α=5% 时 q 落在原子上、严格超越率 "
                         f"{sum(fa5)/len(fa5):.3f}（0 告警，保守），α=10% 时 q "
                         f"移出原子、超越率 {sum(fa10)/len(fa10):.3f} 命中。"
                         "这是计划 §8.2 “阈值极大几乎不报警=无用解”的受控体现，"
                         "与 S1 check6 的并列/原子保守规则一致；native 图无此"
                         "原子（9 单元读数正常）。")
            lines.append("")

    det5 = defaultdict(lambda: defaultdict(list))
    for r in det:
        det5[(r["structure"], r["profile"], r["mechanism"])][
            r["calibrator"]].append(fnum(r["exceed"]))
    lines.append("## S3 检出率（α=5% 冻结阈值；总算子漂移，非纯方差机制）")
    lines.append("")
    lines.append("| 单元 | 机制 | node 校准 power | edge-iid power |")
    lines.append("|---|---|---:|---:|")
    gate_b2 = None
    strong_mechs = ("M1_x3.0", "M2_hubs_x4.0")
    for key, byc in sorted(det5.items()):
        n_ = sum(byc.get("node", [])) / max(len(byc.get("node", [])), 1)
        e_ = sum(byc.get("edge-iid", [])) / max(len(byc.get("edge-iid", [])), 1)
        if key[0] in ("cora-cc", "dblp-ca") and key[2] in strong_mechs:
            # collapse = strong mechanisms also stay at nominal level
            gate_b2 = (False if gate_b2 is None else gate_b2) or n_ > 0.10
        lines.append(f"| {key[0]}/{key[1]} | {key[2]} | {n_:.3f} | {e_:.3f} |")
    lines.append("")
    if s3_available:
        lines.append(f"- **Gate B2**（强替代机制 M1_x3/M2_hubs 下检出未崩溃，"
                     f"power>10%）：**{'达成' if gate_b2 else '未达成'}**"
                     f"（M1/M2 为均值+方差混合漂移，按 §8.3 记为漂移检测；"
                     f"弱信号 M2_lowdeg ≈ 名义水平（该不响应是否“正确”需 S9c 匹配边缘机制的回答，见下））")
        lines.append("- **Gate B3**（相对整次运行基线的增益）：见 S7 段——"
                     "已测运行池支持将 run-pool 作为实际基线；node-fit 零告警；"
                     "node 校准价值在指定模型保证与失配诊断。")
    else:
        lines.append("- Gate B2/B3：待 S3 数据。")
    lines.append("")

    inf = json.load(open(RESULTS / "influence_residual_summary.json"))
    lines.append("## S4 归一化影响与剩余项（T1）")
    lines.append("")
    lines.append("| 结构 | X2_F(S) | Σv_u | 剩余占比 | E‖ΣF_u‖₂ | E‖X‖₂ | raw: E‖ΣF_u‖₂ / E‖X_B‖₂ | 残差检查 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in inf:
        resid = ("noise-floor " + f"{r['resid_cond_maxabs']:.3f}"
                 if r["full_node"] else "subset")
        share = (1 - r["sum_vS"] / r["X2_frob_S"]) if r["full_node"] else float("nan")
        lines.append(
            f"| {r['structure']}/{r['profile']} | {r['X2_frob_S']:.3f} | "
            f"{r['sum_vS'] if r['full_node'] else float('nan'):.3f} | "
            f"{share:.3f} | {r['E_spec_sumF']:.3f} | {r['E_spec_X']:.3f} | "
            f"{r['E_spec_sumFB']:.3f} / {r['E_spec_XB']:.3f} | {resid} |")
    lines.append("")
    lines.append("- native 图为 top-影响 + 随机背景节点子集（不得称为全体能量）；")
    lines.append("- 谱层面：`E‖ΣF_u‖₂` 对比 `E‖X‖₂`，raw 同列对照 —— 归一化后"
                 "相干放大是否保留、平坦抵消是否仍近似成立。")
    # Gate A4: same-degree influence contrast (family hubs, degree 120)
    try:
        nrows = read_csv(RESULTS / "influence_residual.csv")
        hub_rows = {}
        for r in nrows:
            if r["node"] == "0":
                hub_rows[r["profile"]] = r["v_u_S"]

        def vfmt(x):
            return f"{float(x):.3f}"
        if len(hub_rows) >= 2:
            lines.append("")
            lines.append(f"- **同度对比（deg=120 hub）**：相干块 v_u="
                         f"{vfmt(hub_rows.get('fixed', 'nan'))}，"
                         f"平坦抵消点 v_u={vfmt(hub_rows.get('gamma=0.05_cancel', 'nan'))}，"
                         f"最大竞争 v_u={vfmt(hub_rows.get('gamma=0.95_max', 'nan'))}；"
                         "度数相同、v_u 相差数量级 → Gate A4：影响指标携带"
                         "度数之外的结构信息（spearman(v,deg) 见 JSON）。")
    except FileNotFoundError:
        pass
    lines.append("")

    theo = json.load(open(RESULTS / "theory_checks.json"))
    lines.append("## S5 T4/T2 数值核验")
    lines.append("")
    t4 = theo["t4"]
    lines.append("| 结构 | T4 违例 | max LHS/RHS |")
    lines.append("|---|---|---|")
    for name, v in t4.items():
        lines.append(f"| {name} | {v['violations']} | "
                     f"{v['max_lhs_over_rhs']:.3f} |")
    lines.append("")
    lines.append("| 结构 | V_ES_hat ± se | X2_F | V/X2 | 有效样本量 | 尾检查 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for name, v in theo["es"].items():
        if "se_V" in v:
            trivial = sum(t["trivially_true"] for t in v["tail"])
            tail_txt = (f"{trivial}/{len(v['tail'])} 平凡成立(bound≥1)，"
                        "不构成验证")
            lines.append(f"| {name} | {v['V_ES_hat']:.2f} ± {v['se_V']:.2f} | "
                         f"{v['X2_frob']:.2f} | {v['ratio_V_over_X2']:.2f} | "
                         f"{v['effective_n']:.0f}/{v['n_replacements']} | "
                         f"{tail_txt} |")
        else:  # 兼容旧字段
            tail_ok = all(t.get("valid", True) for t in v["tail"])
            lines.append(f"| {name} | {v['V_ES_hat']:.4f} | "
                         f"{v['X2_frob']:.4f} | {v['ratio_V_over_X2']:.2f} | "
                         f"– | {'CHECK' if tail_ok else 'CHECK'} |")
    lines.append("")
    lines.append("- V_ES 点估计；**成为证书需上置信界（T2b）**，当前不得称证书。")
    lines.append("")

    # ---- Gate C: analytic bound usability -----------------------------
    lines.append("## Gate C 理论可用性判定")
    lines.append("")
    try:
        ucb = json.load(open(RESULTS / "es_ucb.json"))
    except FileNotFoundError:
        ucb = {}
    vacuous = []
    for name, u in ucb.items():
        t95 = u["thresholds"][0] if u["thresholds"] else None
        vacuous.append(f"{name}: V̂={u['V_hat']:.1f} vs X2={u['X2_frob']:.1f}"
                       + (f"，95%截断丢尾占比 {t95['tail_share_of_mean']:.2f}"
                          if t95 else ""))
    lines.append("- T4 工程引理：全结构 0 违例，但实测 LHS/RHS ≤ 0.25（native "
                 "≤ 0.009）——可用于定位与保守控制，**不足以作为非空泛主界**。")
    lines.append("- T2 Efron–Stein 路径：" + "；".join(vacuous) + "。")
    lines.append("- 判定（计划 §12 Gate C）：**解析上界目前未达到主贡献地位**"
                 "（native 非空泛性未达成，V̂/X2 甚至 <1 说明估计不可靠、"
                 "尾界 V_ES/t² 在实测分位点 ≥1）；实际校准因此建立在秩/"
                 "交换性保证上（S3）。T2b（V_ES 上置信界）与 T3（一阶项矩阵 "
                 "Bernstein + 剩余项阶）保持为开放研究任务，不包装成已得结果。")
    lines.append("")
    lines.append("## Gate D 扩展 B（风险降权）：未启动")
    lines.append("")
    lines.append("- 前置条件（A 显示可解释的归一化风险差异）已满足"
                 "（S2 native ratio_S>1 且 S4 同度 v_u 跨数量级），但按计划"
                 "须先补 A3/S8 与 B3；v_u 已可计算（influence_residual.csv），"
                 "降权接口随时可用。")
    lines.append("")
    lines.append("## S6 L2 拟合节点律（T5 原型，Cora）")
    lines.append("")
    l2p = RESULTS / "l2_fitted_law.csv"
    if l2p.exists():
        l2 = read_csv(l2p)
        agg3 = defaultdict(list)
        for r in l2:
            agg3[(fnum(r["alpha"]), r["eval"])].append(fnum(r["exceed"]))
        lines.append("| α | 拟合模型内 | 真律下 | 平均 TV(r,r̂) |")
        lines.append("|---|---:|---:|---:|")
        for alpha in (0.01, 0.05, 0.10):
            fh = sum(agg3[(alpha, "fitted-model")]) / max(
                len(agg3[(alpha, "fitted-model")]), 1)
            ft = sum(agg3[(alpha, "true-law")]) / max(
                len(agg3[(alpha, "true-law")]), 1)
            mtv = next(fnum(r["mean_tv"]) for r in l2
                       if fnum(r["alpha"]) == alpha)
            lines.append(f"| {alpha:.2f} | {fh:.3f} | {ft:.3f} | {mtv:.4f} |")
        lines.append("")
        lines.append("- 拟合律（独立 2000 批经验频率）下校准的真律超越率与"
                     "标称相容 → L2 在均匀弱误差律下失配代价小；TV 粗界"
                     "（T5 空泛风险）按原样保留为开放问题，不据此声称覆盖"
                     "真实训练依赖（等级 4/5）。")
    else:
        lines.append("- 待生成。")
    lines.append("")

    l3p = RESULTS / "l3_run_pool.csv"
    lines.append("## S7/A3-S8 L3 整次运行校准（真实预测器机制矩阵：LP / "
                 "prototype / GCN5f crossfit）")
    lines.append("")
    if l3p.exists():
        l3 = read_csv(l3p)
        lines.append("| 数据 | 预测器(机制) | arm | α | 评估池超越率 [95% CI] | 真实训练次数 | 秒 |")
        lines.append("|---|---|---|---|---|---:|---:|")
        MECH = {"LP": "标签集随机", "proto": "半split随机",
                "gcn5f": "5折crossfit", "gcn5f-e120": "crossfit(e120)"}
        for r in l3:
            lines.append(
                f"| {r['tag']} | {r['predictor']}"
                f"({MECH.get(r['predictor'], '?')}) | {r['arm']} | "
                f"{fnum(r['alpha']):.2f} | "
                f"{fnum(r['exceed_on_eval_runs']):.3f} "
                f"[{fnum(r['cp_lo']):.3f},{fnum(r['cp_hi']):.3f}] | "
                f"{r['n_runs_train']} | {r['secs_train']} |")
        lines.append("")
        lines.append("- 已测 run-pool 的 α=5%/10% 点估计及条件区间见表；"
                     "这是有限样本读数，不主张精确命中目标。个别 α=1% 单元"
                     "校准预算不足 q=∞，该设置无检出能力，单独标注。")
        lines.append("- node-fit 在**所有**真实运行机制下零告警（保守未报，非误报）（包括 GCN "
                     "5-fold crossfit——run 间预测差异 10.5%）：乘积律重采样的"
                     "分数分布系统性**高于**真实运行分数分布，阈值过保守、0 "
                     "告警。方向与计划 §3.4 预警一致（失配是保守/零告警方向，"
                     "不是误报方向）；L1 下 edge-iid 的过度超越与 L3 下 "
                     "node-fit 的保守零告警**不可混用为同一叙事**。")
        lines.append("- **Gate B3 判定**：整次运行分位数作为本研究 L3 的主要实际基线；"
                     "node 校准的价值在层级保证 + 失配诊断 + 影响结构解释"
                     "（§8.2 power 读数），不是 run-pool 的省钱替代。"
                     "成本证据：LP 250 训练 2.5 秒；GCN 120 训练 46 分钟——"
                     "预算权衡故事成立但结论方向不变。"
                     "DBLP GCN 运行池成本外推 4–12 小时，移入批量窗口"
                     "（followups F3），不阻塞判定。")
    else:
        lines.append("- 待生成（run scripts/s7_run_pool.py）。")
    lines.append("")

    # ---- S9: review-round-2 experiments ----
    s9a_p = RESULTS / "normalization_response.csv"
    if s9a_p.exists():
        s9a = read_csv(s9a_p)
        s2_map = {r["structure"] + "/" + r["profile"]: r["ratio_S"]
                  for r in pilot}
        lines.append("## S9a 归一化抵消的线性响应分解（评审改造一）")
        lines.append("")
        lines.append("在独立参考中心 p 处 S(a)-S0 = J_A(E)+resid，"
                     "J = 关系项 R1 + 度抵消项 Rdeg（C = D0^{-1}diag(E1)）；"
                     "两臂在同一冻结基点上测（共同固定中心口径）。")
        lines.append("")
        lines.append("| 结构 | ratio_R1 | ratio_J | ratio_true | S2 ratio_S | 抵消node | 抵消edge | resid/true node |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in s9a:
            key = r["structure"] + "/" + r["profile"]
            lines.append(
                f"| {key} | {fnum(r['ratio_R1']):.2f} | "
                f"{fnum(r['ratio_J']):.2f} | {fnum(r['ratio_true']):.2f} | "
                f"{fnum(float(s2_map.get(key, 'nan'))):.2f} | "
                f"{fnum(r['node_canc_frac']):.2f} | "
                f"{fnum(r['edge_canc_frac']):.2f} | "
                f"{fnum(r['node_resid_frac']):.2f} |")
        lines.append("")
        lines.append("- **辅助导数通道分析**："
                     "ratio_R1（固定度加权后的关系项）已把 coherent 的 raw 2.93 "
                     "压到 ~1.1——第一层抵消来自静态 D0^{-1/2} 加权（S2 中 "
                     "ratio_S0=1.16 与此一致）；动态度修正项 Rdeg 是第二层，"
                     "对相干块再压（canc_frac 0.4-0.5，familyA ratio_J≈0.45），"
                     "对原生图净抵消弱甚至轻微增强（canc_frac≈0）——"
                     "这些读数显示局部导数通道随结构改变，不单独解释S2波动比反转。")
        lines.append("- **判定（降级版，按评审第三轮）**：静态度加权显著改变"
                     "放大程度；线性响应揭示了度修正的抵消通道；但有限幅度"
                     "随机扰动仍受显著高阶项支配——familyA 上 ratio_J=0.45 与"
                     "同基点 ratio_true=1.29 分处 1 的两侧，方向并不一致，"
                     "resid/true>1。当前数据不能写成“已解释真实归一化后的"
                     "反转”；历史 S10a 范数差下界也不能认证有效范围。")
        lines.append("- 注：表中 cos 列为 Frobenius 内积除以谱范数乘积的粗指标"
                     "（可超 ±1），负值表示关系项与度抵消项方向相反（抵消）。")
        lines.append("")

    s9b_p = RESULTS / "directional_detection.csv"
    if s9b_p.exists():
        s9b = read_csv(s9b_p)
        lines.append("## S9b 方向标准化检测 T_g = max_j |v_j^T(S-M_hat)v_j|/"
                     "(sigma_j+eps)（评审改造二）")
        lines.append("")
        lines.append("inf/var/rand组在参考阶段冻结，目标alpha=5%；实际正常误报率见normal行，"
                     "不假定每个冻结阈值都精确达到5%。本表是方向生成修复前的历史探索。")
        lines.append("")
        lines.append("| 数据 | 机制 | norm | inf | var | rand |")
        lines.append("|---|---|---:|---:|---:|---:|")
        piv = defaultdict(dict)
        for r in s9b:
            piv[(r["structure"], r["mechanism"])][r["statistic"]] =                 fnum(r["rate" if "rate" in r else "power"])
        for k, v in sorted(piv.items()):
            fmt = lambda g: f"{v.get(g, float('nan')):.3f}"
            lines.append(f"| {k[0]} | {k[1]} | {fmt('norm')} | {fmt('inf')} "
                         f"| {fmt('var')} | {fmt('rand')} |")
        lines.append("")
        by_str = defaultdict(list)
        for r in s9b:
            if r["mechanism"] not in ("normal",):
                by_str[r["structure"]].append(r)
        for st, rr in sorted(by_str.items()):
            alt = [x for x in rr if x["mechanism"].startswith(("M1", "M2"))]
            wins = sum(1 for x in alt if x["statistic"] == "inf" and
                       fnum(x["rate"]) > max(
                           [fnum(y["rate"]) for y in alt
                            if y["statistic"] == "norm"
                            and y["mechanism"] == x["mechanism"]]
                           or [0]))
            mechanisms = len({x["mechanism"] for x in alt})
            lines.append(f"- **{st} 描述**：漂移机制 {mechanisms} 格中 "
                         f"inf>norm 有 {wins} 格。"
                         "只报告点估计次序；未做配对差异确认，不把图间差异归因为hub集中度。")
        lines.append("- 未建立方向统计的稳定优势。rand为有效竞争读数，均值与波动"
                     "都可能进入响应；hub集中度解释仅为假设。方向生成修复后需要"
                     "重新冻结并复跑，S9c也不提供选择方向优越性的普遍保证。")
        lines.append("")

    s9c_p = RESULTS / "dependence_dial.csv"
    if s9c_p.exists():
        s9c = read_csv(s9c_p)
        lines.append("## S9c 匹配边缘依赖拨盘（逐边边缘 p_e 严格不变、只改依赖结构；"
                     "动态归一化后均值与波动可同变，非纯波动机制）")
        lines.append("")
        for base in sorted({r.get("baseline", "cal_A_x1.00") for r in s9c}):
            lines.append(f"**基线 {base}**")
            lines.append("")
            lines.append("| 数据 | x | norm | inf | var | rand | 漂移vsA |")
            lines.append("|---|---|---:|---:|---:|---:|---:|")
            piv2 = defaultdict(dict)
            drift = {}
            for r in [rr for rr in s9c if rr.get("baseline", "cal_A_x1.00") == base]:
                piv2[(r["structure"], r["mechanism"])][r["statistic"]] =                     fnum(r["power"])
                drift[(r["structure"], r["mechanism"])] = fnum(
                    r["mean_drift_vs_A"])

            def _key(it):
                try:
                    return (it[0][0], -float(it[0][1].replace("dep_x", "")))
                except ValueError:
                    return (it[0][0], 0)
            for k, v in sorted(piv2.items(), key=_key):
                fmt = lambda g: f"{v.get(g, float('nan')):.3f}"
                lines.append(f"| {k[0]} | {k[1].replace('dep_x', 'x=')} | "
                             f"{fmt('norm')} | {fmt('inf')} | {fmt('var')} "
                             f"| {fmt('rand')} | {drift[k]:.3f} |")
            lines.append("")
        lines.append("- **判定（按评审第三轮修正）**：inf 组响应高度依赖参考"
                     "机制——cal_A 基线上平坦（不响应依赖丧失），但 "
                     "cal_B/cal_mix 基线检测依赖出现时原生结构有明确"
                     "响应（cora 24.8–31.0%、dblp 36.4–37.2%，远超"
                     "名义 5%），同时刻 norm 为 87.2–100%——"
                     "**inf 有响应但未稳定优于其他统计量；既未成为强"
                     "方法，也未被普遍否定**。family 合成族上 inf≈0–10%"
                     "（抵消结构方向无能量、符合理论预期），norm 同低。")
        lines.append("- 均值成分说明：拨盘保持逐边边缘，但动态归一化下 "
                     "E S_mix(x) ≠ E S_A——mean_drift_vs_A 列即证据"
                     "（0.016→0.542）。共同冻结中心是合理监测协议，"
                     "但不排除均值成分；纯波动研究需各分布自身中心化"
                     "的对照（列入后续）。")

    # ---- S10: review round 3 ----
    s10a_p = RESULTS / "amplitude_scan.csv"
    if s10a_p.exists():
        s10a = read_csv(s10a_p)
        lines.append("## S10a 历史幅度网格 A(t)=A0+tE（共同基点响应与范数差下界）")
        lines.append("")
        lines.append("| 结构 | 范数差代理跨 0.5 的网格 t | ratio 穿越 1 的网格 t | "
                     "ratio(t=0.05) | ratio(t=0.5) | ratio(t=1) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        byst = defaultdict(list)
        cross = {}
        for r in s10a:
            if r["t"] == "T50_resid":
                cross[r["structure"] + "/" + r["profile"]] = (
                    r["resid_frac_node"], r["ratio_t"])
            else:
                byst[r["structure"] + "/" + r["profile"]].append(r)
        for k, rr in byst.items():
            rr.sort(key=lambda x: float(x["t"]))

            def rt(t):
                c = [x for x in rr if abs(float(x["t"]) - t) < 0.026]
                return float(c[0]["ratio_t"]) if c else float("nan")
            t50, x1 = cross.get(k, ("", ""))
            lines.append(
                f"| {k} | {t50} | {x1} | "
                f"{rt(0.05):.2f} | {rt(0.5):.2f} | {rt(1.0):.2f} |")
        lines.append("")
        lines.append("- **度量修复**：历史 `resid_frac_node` 实际计算 "
                     "E|‖ΔS‖₂−‖tJ‖₂| / E‖ΔS‖₂，而非矩阵剩余项 "
                     "E‖ΔS−tJ‖₂ / E‖ΔS‖₂。反三角不等式只保证前者≤后者；"
                     "代理跨 0.5 可以排除该点的良好一阶近似，代理低于 0.5 "
                     "不能认证有效范围。撤回原 t≈0.2–0.4 的有效区间断言。")
        lines.append("- 共同基点下比值的方向与各臂自身中心化波动（S2）不同，"
                     "不称为同一反转的因果解释；网格穿越只定位到相邻采样幅度之间。"
                     "历史 DBLP 中段曾有手工数值复核，未保存完整逐样本数值诊断；"
                     "不能据此断言 S2/S3/S9 均未受影响。修复后的脚本将实际剩余项"
                     "与求解残差写入 `amplitude_scan_v2.csv`，需另行重跑；"
                     "本轮保留历史测量，S10a 不支撑主贡献。")
        lines.append("")

    s10b_p = RESULTS / "two_sided_detection.csv"
    if s10b_p.exists():
        s10b = read_csv(s10b_p)
        lines.append("## S10b 双侧校准检测 p_two = 2·min(p_up, p_lo)"
                     "（同校准集秩、并列保守）")
        lines.append("")
        lines.append("| 结构 | 机制 | stat | 单侧 | 双侧 |")
        lines.append("|---|---|---|---:|---:|")
        piv3 = defaultdict(dict)
        for r in s10b:
            piv3[(r["structure"], r["mechanism"], r["statistic"])][
                r["test"]] = fnum(r["reject"])
        for k, v in sorted(piv3.items()):
            lines.append(f"| {k[0]} | {k[1]} | {k[2]} | "
                         f"{v.get('one', float('nan')):.3f} | "
                         f"{v.get('two', float('nan')):.3f} |")
        lines.append("")
        fpr = [r for r in s10b if r["mechanism"] == "normal_FPR_check"]
        f2 = max((fnum(r["reject"]) for r in fpr
                  if r["test"] == "two"), default=float("nan"))
        lines.append(f"- 正常机制双侧 FPR 最大 {f2:.3f}（标称 0.05）；"
                     "仅为当前冻结阈值的经验读数，不能由最大点估计证明校准。"
                     "双侧有效性来自两侧有效秩 p 值与并集界。")
        lines.append("- **判读**：依赖丧失拨盘位点上 cora 的 norm “零告警”"
                     "**主要是单侧方向伪影**（one=0.000 → two≈0.99），评审"
                     "第五点被数据证实；但 familyA 上 norm 双侧仍为 0（one 与"
                     "two 均 0.000）——其分数分布被 hub 翻转原子控制，B 臂"
                     "分数落在校准 IQR 内，双侧化救不了原子遮蔽。两种机制"
                     "（方向伪影 vs 原子遮蔽）必须分开陈述，检测能力评估"
                     "需先看参考分数分布的形状。两点补充（DBLP）：漂移机制下"
                     "非-rand 统计中 inf 最强（M2_hubs 26.7% vs norm 13.3%，"
                     "与 S9b 一致），双侧轻微稀释（下尾无证据也占概率质量）；"
                     "拨盘双侧化后 inf 也显出下尾响应（x=0 时 28.0%）——与"
                     "评审“inf 有响应但不稳定占优”的表述一致。rand 对均值"
                     "成分持续高响应（有效竞争结果，目标相关）。")
        lines.append("")
        lines.append("")

    # ---- S11/S12: review round-4 closing items ----
    s11_p = RESULTS / "l3_fixed_split.csv"
    if s11_p.exists():
        s11 = read_csv(s11_p)
        lines.append("## S11 固定训练划分、只改变初始化（L3 机制隔离）")
        lines.append("")
        lines.append("半监督 GCN（固定 20% 标注、hidden64/epochs500(Cora)"
                     "或120(DBLP)/lr0.2），run 随机性只来自初始化。")
        lines.append("")
        lines.append("| 数据 | arm | α | 评估池超越率 [95% CI] | 训练次数 |")
        lines.append("|---|---|---|---|---:|")
        for r in s11:
            lines.append(f"| {r['tag']} | {r['arm']} | "
                         f"{fnum(r['alpha']):.2f} | "
                         f"{fnum(r['exceed_on_eval_runs']):.3f} "
                         f"[{fnum(r['cp_lo']):.3f},{fnum(r['cp_hi']):.3f}] | "
                         f"{r['n_runs_train']} |")
        lines.append("")
        lines.append("- 判读：两个固定划分设置中，run-pool 条件区间均包含 α，"
                     "但样本少（Cora 评估 60 次、DBLP 40 次），不构成等效性证明。"
                     "Cora α=5% 1/60、α=10% 9/60；DBLP 两档 1/40。"
                     "node-fit 均为 0 次超越；不能由 0 次观测推断总体超越率为零。"
                     "区间宽度与评估样本量、条件超越概率有关，不能归因于分数分布窄。")
        lines.append("")

    s12_p = RESULTS / "confirm_results.csv"
    if s12_p.exists():
        s12 = summarize_confirmation(read_csv(s12_p))
        lines.append("## S12 新种子复核（修正区间与结论范围）")
        lines.append("")
        lines.append("| 主张 | 结构 | 指标 | 估计 [95% CI] | 次数 | 区间方法 |")
        lines.append("|---|---|---|---|---:|---|")
        for r in s12:
            ci = f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
            lines.append(f"| {r['claim']} | {r['structure']} | "
                         f"{r['metric']} | {fnum(r['estimate']):.3f} {ci} | "
                         f"{r['count']}/{r['n']} | {r['interval_method']} |")
        lines.append("")
        lines.append("- C1：112/3000=3.73%；三批使用不同参考中心与阈值，"
                     "原合并二项 CP 区间撤销。给定三套冻结工件后，独立有界评估"
                     "的 Hoeffding 95% 区间为 [1.25%,6.21%]，对象是三批条件"
                     "超越概率的样本量加权均值，不是校准重抽样的边际覆盖。"
                     "不能写精确命中 5%，也不能从原区间判定秩保证失败。")
        lines.append("- C2：DBLP inf 31.6%、norm 24.0%，为两组不同观测；"
                     "缺少配对结果与本轮正常机制误报检查。仅保留描述性差异，"
                     "不主张确认了检测优势。C3：双侧 394/400、单侧 0/400，"
                     "上述区间已补齐；检验选择仍为辅助诊断。")
        lines.append("- S12 未新增 L3 训练运行，不能作为真实预测器独立确认。"
                     "脚本记录了另设种子，但未保存运行前带时间戳的冻结协议，"
                     "因此称新种子复核，不追认预注册确认。")
        lines.append("")

    lines.append("## 判定汇总")
    lines.append("")
    if s3_available:
        a_ok = bool(gate_a1) and bool(gate_a2)
        lines.append("- **Gate A（主线继续）："
                     f"{'可继续' if a_ok else '停止归一化谱风险校准路线'}**"
                     f"（A1={bool(gate_a1)}，A2={bool(gate_a2)}，A4 见 S4 段）")
        b_partial = ("node 校准恢复目标（A2）+ 检出未崩溃（B2）= "
                     + ("初步支持继续" if (gate_a2 and gate_b2) else "不足"))
        lines.append(f"- **Gate B（方法价值）：{b_partial}。"
                     "B3：run-pool 是已测 LP/proto/GCN 机制的主要实际基线，"
                     "node-fit 均观察到零超越——论文定位为“层级保证 + "
                     "机制失配诊断 + 影响结构解释”，不主张 node 校准是 "
                     "run-pool 的预算替代（详见 S7 段）。")
    else:
        lines.append("- Gate A/B 判定：待 S3 完成后重跑本脚本生成。")
    lines.append("- 后续围绕单一主线：按 claims_evidence.md 的三条主张整合稿件；"
                 "若要将真实运行复核升级为独立确认，先保存前瞻协议、运行预测缓存"
                 "和逐分数数值诊断，再用新种子检验。方向检测、新数据、"
                 "T2b/T3 与 Gate D 均不作为本轮收尾的新增主贡献。")

    out = HERE / "docs" / "gate_ab_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out)


def rows_by(key, rows, cname):
    return [r for r in rows if r["calibrator"] == cname]


if __name__ == "__main__":
    main()
