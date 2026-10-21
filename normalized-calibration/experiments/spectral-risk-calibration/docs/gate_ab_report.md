# 第一篇证据报告：归一化影响、分层校准与运行失配诊断

生成脚本：`scripts/build_gate_report.py`；数据：`results/*.csv|json`。
内部投入标准（15% 线为筛选线，非理论界、非发表标准）。

**单一主线**：节点诱导依赖如何改变归一化谱风险，以及校准需要保留哪一层随机律。
主证据为 S1–S4、S6–S8、S11 与 S12-C1；S9/S10 和 S12-C2/C3 仅作辅助诊断。
本报告读取历史CSV；方向生成与矩阵剩余项的代码修复尚未全量重跑，未来v2结果另存且不与旧表混用。

**中心分工**：S2 以各臂自身均值中心化（||S−ES_arm||）；
S9a/S10a 是共同确定性基点 S(p) 的扰动响应，并非各臂中心化波动；
监测/校准 S3/S6/S7/S8/S9b/S9c/S10b/S11/S12 使用各单元冻结的 M_hat。
**保证分工**：秩保证条件于独立参考阶段，对校准集与新观测联合取概率；
固定一次阈值的条件超越率可以高于或低于 α。CP 区间只描述该条件率；
区间覆盖 α 不证明等效，点估计低于 α 不等于校准失败。

## S1 代数与数值验证

- all_pass = **True**
- 归一化一阶分解在小图上精确成立：max|E[R|Y_u=z]| = 2.8e-15；能量恒等式相对缺口 0.0e+00
- 固定度两臂均值精确相等（gap=1.1e-15）；动态度均值移位非零 ||E S_A − E S_B||₂ = 0.046

## S2 三算子对照（B / S0 / S）

| 结构 | 误差律 | ratio_raw | ratio_S0 | ratio_S | 均值移位 ||ES_A−ES_B|| | shift_scale |
|---|---|---:|---:|---:|---:|---:|
| familyA_coherent | fixed | 2.93 | 1.16 | 0.24 | 0.5323 | 4.280 |
| familyB_flat | gamma=0.05_cancel | 1.01 | 1.00 | 1.00 | 0.0175 | 0.036 |
| familyB_flat | gamma=0.5 | 1.40 | 1.02 | 0.99 | 0.0882 | 0.185 |
| familyB_flat | gamma=0.95_max | 2.28 | 0.85 | 0.53 | 0.4370 | 1.721 |
| cora-cc | binary-eps0.05 | 3.25 | 1.39 | 1.24 | 0.0838 | 0.145 |
| cora-cc | binary-eps0.15 | 2.93 | 1.41 | 1.09 | 0.1815 | 0.314 |
| cora-cc | multiclass-eps0.05 | 3.47 | 1.24 | 1.26 | 0.0841 | 0.148 |
| cora-cc | multiclass-eps0.15 | 3.28 | 1.29 | 1.11 | 0.1727 | 0.300 |
| cora-cc | hetero-0.02/0.30 | 3.23 | 1.45 | 1.05 | 0.2266 | 0.396 |
| dblp-ca | binary-eps0.05 | 2.48 | 1.70 | 1.13 | 0.1894 | 0.265 |
| dblp-ca | binary-eps0.15 | 2.16 | 1.42 | 1.03 | 0.2918 | 0.417 |
| dblp-ca | hetero-0.02/0.30 | 2.15 | 1.77 | 1.07 | 0.3361 | 0.486 |

## S3 模型内校准（α=5%，批次均值 + 逐批区间端点包络）

方括号为各批 CP 区间的最小下端点和最大上端点，不是合并样本的 95% CI；PASS 是历史内部筛选结果，不是显著性或等效性检验。

| 单元 | node 校准 | edge-iid 校准 | 判定 |
|---|---|---|---|
| cora-cc/binary-eps0.05 | 0.052 [0.037,0.072] | 0.995 [0.983,1.000] | PASS |
| cora-cc/binary-eps0.15 | 0.050 [0.036,0.067] | 0.977 [0.956,0.994] | PASS |
| cora-cc/hetero-0.02/0.30 | 0.059 [0.030,0.093] | 0.475 [0.405,0.549] | PASS |
| cora-cc/multiclass-eps0.05 | 0.048 [0.033,0.065] | 0.997 [0.987,1.000] | PASS |
| cora-cc/multiclass-eps0.15 | 0.055 [0.029,0.083] | 0.990 [0.975,0.999] | PASS |
| dblp-ca/binary-eps0.05 | 0.046 [0.014,0.093] | 1.000 [0.988,1.000] | PASS |
| dblp-ca/binary-eps0.15 | 0.051 [0.014,0.101] | 0.968 [0.907,0.998] | PASS |
| dblp-ca/hetero-0.02/0.30 | 0.039 [0.009,0.089] | 0.997 [0.971,1.000] | PASS |
| familyA_coherent/fixed | 0.000 [0.000,0.002] | 0.092 [0.078,0.107] | CHECK |

- **Gate A1**（独立边在 α=5% 明显过度超越 >15%）：**达成**
- **Gate A2**（已测原生图满足内部相容筛选线）：**达成**
### α 敏感性（均值超越率）

| 单元 | α | node | edge-iid |
|---|---|---:|---:|
| cora-cc/binary-eps0.05 | 0.01 | 0.010 | 0.941 |
| cora-cc/binary-eps0.05 | 0.05 | 0.052 | 0.995 |
| cora-cc/binary-eps0.05 | 0.10 | 0.102 | 0.999 |
| cora-cc/binary-eps0.15 | 0.01 | 0.017 | 0.841 |
| cora-cc/binary-eps0.15 | 0.05 | 0.050 | 0.977 |
| cora-cc/binary-eps0.15 | 0.10 | 0.097 | 0.998 |
| cora-cc/hetero-0.02/0.30 | 0.01 | 0.017 | 0.329 |
| cora-cc/hetero-0.02/0.30 | 0.05 | 0.059 | 0.475 |
| cora-cc/hetero-0.02/0.30 | 0.10 | 0.107 | 0.601 |
| cora-cc/multiclass-eps0.05 | 0.01 | 0.014 | 0.973 |
| cora-cc/multiclass-eps0.05 | 0.05 | 0.048 | 0.997 |
| cora-cc/multiclass-eps0.05 | 0.10 | 0.106 | 1.000 |
| cora-cc/multiclass-eps0.15 | 0.01 | 0.008 | 0.556 |
| cora-cc/multiclass-eps0.15 | 0.05 | 0.055 | 0.990 |
| cora-cc/multiclass-eps0.15 | 0.10 | 0.109 | 0.997 |
| dblp-ca/binary-eps0.05 | 0.01 | 0.001 | 0.762 |
| dblp-ca/binary-eps0.05 | 0.05 | 0.046 | 1.000 |
| dblp-ca/binary-eps0.05 | 0.10 | 0.099 | 1.000 |
| dblp-ca/binary-eps0.15 | 0.01 | 0.004 | 0.789 |
| dblp-ca/binary-eps0.15 | 0.05 | 0.051 | 0.968 |
| dblp-ca/binary-eps0.15 | 0.10 | 0.099 | 0.993 |
| dblp-ca/hetero-0.02/0.30 | 0.01 | 0.002 | 0.638 |
| dblp-ca/hetero-0.02/0.30 | 0.05 | 0.039 | 0.997 |
| dblp-ca/hetero-0.02/0.30 | 0.10 | 0.074 | 1.000 |

- **familyA 原子性说明（非 bug，结构性质）**：相干族的分数尾部由 hub 翻转离散状态主导，最大值处存在约 10% 质量的原子；α=5% 时 q 落在原子上、严格超越率 0.000（0 告警，保守），α=10% 时 q 移出原子、超越率 0.054 命中。这是计划 §8.2 “阈值极大几乎不报警=无用解”的受控体现，与 S1 check6 的并列/原子保守规则一致；native 图无此原子（9 单元读数正常）。

## S3 检出率（α=5% 冻结阈值；总算子漂移，非纯方差机制）

| 单元 | 机制 | node 校准 power | edge-iid power |
|---|---|---:|---:|
| cora-cc/binary-eps0.05 | M1_x1.5 | 0.079 | 1.000 |
| cora-cc/binary-eps0.05 | M1_x2.0 | 0.102 | 1.000 |
| cora-cc/binary-eps0.05 | M1_x3.0 | 0.174 | 1.000 |
| cora-cc/binary-eps0.05 | M2_hubs_x4.0 | 0.206 | 1.000 |
| cora-cc/binary-eps0.05 | M2_lowdeg_x4.0 | 0.046 | 0.997 |
| cora-cc/binary-eps0.15 | M1_x1.5 | 0.070 | 0.989 |
| cora-cc/binary-eps0.15 | M1_x2.0 | 0.095 | 1.000 |
| cora-cc/binary-eps0.15 | M1_x3.0 | 0.132 | 0.999 |
| cora-cc/binary-eps0.15 | M2_hubs_x4.0 | 0.098 | 1.000 |
| cora-cc/binary-eps0.15 | M2_lowdeg_x4.0 | 0.059 | 0.972 |
| cora-cc/hetero-0.02/0.30 | M1_x1.5 | 0.062 | 0.638 |
| cora-cc/hetero-0.02/0.30 | M1_x2.0 | 0.084 | 0.767 |
| cora-cc/hetero-0.02/0.30 | M1_x3.0 | 0.062 | 0.901 |
| cora-cc/hetero-0.02/0.30 | M2_hubs_x4.0 | 0.060 | 0.554 |
| cora-cc/hetero-0.02/0.30 | M2_lowdeg_x4.0 | 0.057 | 0.488 |
| cora-cc/multiclass-eps0.05 | M1_x1.5 | 0.073 | 1.000 |
| cora-cc/multiclass-eps0.05 | M1_x2.0 | 0.133 | 1.000 |
| cora-cc/multiclass-eps0.05 | M1_x3.0 | 0.198 | 1.000 |
| cora-cc/multiclass-eps0.05 | M2_hubs_x4.0 | 0.293 | 1.000 |
| cora-cc/multiclass-eps0.05 | M2_lowdeg_x4.0 | 0.048 | 0.997 |
| cora-cc/multiclass-eps0.15 | M1_x1.5 | 0.084 | 1.000 |
| cora-cc/multiclass-eps0.15 | M1_x2.0 | 0.108 | 1.000 |
| cora-cc/multiclass-eps0.15 | M1_x3.0 | 0.145 | 1.000 |
| cora-cc/multiclass-eps0.15 | M2_hubs_x4.0 | 0.177 | 1.000 |
| cora-cc/multiclass-eps0.15 | M2_lowdeg_x4.0 | 0.043 | 0.989 |
| dblp-ca/binary-eps0.05 | M1_x1.5 | 0.044 | 1.000 |
| dblp-ca/binary-eps0.05 | M1_x2.0 | 0.056 | 1.000 |
| dblp-ca/binary-eps0.05 | M1_x3.0 | 0.064 | 1.000 |
| dblp-ca/binary-eps0.05 | M2_hubs_x4.0 | 0.082 | 1.000 |
| dblp-ca/binary-eps0.05 | M2_lowdeg_x4.0 | 0.051 | 1.000 |
| dblp-ca/binary-eps0.15 | M1_x1.5 | 0.062 | 0.984 |
| dblp-ca/binary-eps0.15 | M1_x2.0 | 0.062 | 0.996 |
| dblp-ca/binary-eps0.15 | M1_x3.0 | 0.076 | 1.000 |
| dblp-ca/binary-eps0.15 | M2_hubs_x4.0 | 0.089 | 0.998 |
| dblp-ca/binary-eps0.15 | M2_lowdeg_x4.0 | 0.051 | 0.967 |
| dblp-ca/hetero-0.02/0.30 | M1_x1.5 | 0.064 | 1.000 |
| dblp-ca/hetero-0.02/0.30 | M1_x2.0 | 0.042 | 1.000 |
| dblp-ca/hetero-0.02/0.30 | M1_x3.0 | 0.029 | 1.000 |
| dblp-ca/hetero-0.02/0.30 | M2_hubs_x4.0 | 0.056 | 0.998 |
| dblp-ca/hetero-0.02/0.30 | M2_lowdeg_x4.0 | 0.056 | 1.000 |

- **Gate B2**（强替代机制 M1_x3/M2_hubs 下检出未崩溃，power>10%）：**达成**（M1/M2 为均值+方差混合漂移，按 §8.3 记为漂移检测；弱信号 M2_lowdeg ≈ 名义水平（该不响应是否“正确”需 S9c 匹配边缘机制的回答，见下））
- **Gate B3**（相对整次运行基线的增益）：见 S7 段——已测运行池支持将 run-pool 作为实际基线；node-fit 零告警；node 校准价值在指定模型保证与失配诊断。

## S4 归一化影响与剩余项（T1）

| 结构 | X2_F(S) | Σv_u | 剩余占比 | E‖ΣF_u‖₂ | E‖X‖₂ | raw: E‖ΣF_u‖₂ / E‖X_B‖₂ | 残差检查 |
|---|---:|---:|---:|---:|---:|---:|---|
| familyA_coherent/fixed | 2.216 | 2.202 | 0.006 | 0.137 | 0.139 | 6.285 / 6.143 | noise-floor 0.007 |
| familyB_flat/gamma=0.05_cancel | 1.681 | 1.446 | 0.140 | 0.465 | 0.479 | 1.559 / 1.617 | noise-floor 0.040 |
| familyB_flat/gamma=0.95_max | 1.513 | 1.334 | 0.118 | 0.122 | 0.239 | 3.296 / 2.837 | noise-floor 0.073 |
| cora-cc/binary-eps0.05 | 28.241 | nan | nan | 0.610 | 0.712 | 11.228 / 11.819 | subset |
| dblp-ca/binary-eps0.05 | 1274.859 | nan | nan | 0.334 | 0.797 | 2.624 / 6.181 | subset |

- native 图为 top-影响 + 随机背景节点子集（不得称为全体能量）；
- 谱层面：`E‖ΣF_u‖₂` 对比 `E‖X‖₂`，raw 同列对照 —— 归一化后相干放大是否保留、平坦抵消是否仍近似成立。

- **同度对比（deg=120 hub）**：相干块 v_u=1.099，平坦抵消点 v_u=0.000，最大竞争 v_u=1.311；度数相同、v_u 相差数量级 → Gate A4：影响指标携带度数之外的结构信息（spearman(v,deg) 见 JSON）。

## S5 T4/T2 数值核验

| 结构 | T4 违例 | max LHS/RHS |
|---|---|---|
| familyA | 0 | 0.006 |
| familyB_max | 0 | 0.250 |
| cora-cc | 0 | 0.002 |
| dblp-ca | 0 | 0.008 |

| 结构 | V_ES_hat ± se | X2_F | V/X2 | 有效样本量 | 尾检查 |
|---|---:|---:|---:|---:|---|
| familyA | 1.99 ± 0.41 | 2.27 | 0.88 | 23/800 | 3/3 平凡成立(bound≥1)，不构成验证 |
| familyB_max | 1.33 ± 0.30 | 1.58 | 0.84 | 20/800 | 3/3 平凡成立(bound≥1)，不构成验证 |
| cora-cc | 22.29 ± 4.59 | 28.03 | 0.80 | 22/400 | 3/3 平凡成立(bound≥1)，不构成验证 |
| dblp-ca | 1588.85 ± 491.20 | 1282.15 | 1.24 | 10/200 | 3/3 平凡成立(bound≥1)，不构成验证 |

- V_ES 点估计；**成为证书需上置信界（T2b）**，当前不得称证书。

## Gate C 理论可用性判定

- T4 工程引理：全结构 0 违例，但实测 LHS/RHS ≤ 0.25（native ≤ 0.009）——可用于定位与保守控制，**不足以作为非空泛主界**。
- T2 Efron–Stein 路径：familyA: V̂=2.3 vs X2=2.1，95%截断丢尾占比 1.00；cora-cc: V̂=30.8 vs X2=28.0，95%截断丢尾占比 0.75；dblp-ca: V̂=1597.7 vs X2=1278.0，95%截断丢尾占比 0.90。
- 判定（计划 §12 Gate C）：**解析上界目前未达到主贡献地位**（native 非空泛性未达成，V̂/X2 甚至 <1 说明估计不可靠、尾界 V_ES/t² 在实测分位点 ≥1）；实际校准因此建立在秩/交换性保证上（S3）。T2b（V_ES 上置信界）与 T3（一阶项矩阵 Bernstein + 剩余项阶）保持为开放研究任务，不包装成已得结果。

## Gate D 扩展 B（风险降权）：未启动

- 前置条件（A 显示可解释的归一化风险差异）已满足（S2 native ratio_S>1 且 S4 同度 v_u 跨数量级），但按计划须先补 A3/S8 与 B3；v_u 已可计算（influence_residual.csv），降权接口随时可用。

## S6 L2 拟合节点律（T5 原型，Cora）

| α | 拟合模型内 | 真律下 | 平均 TV(r,r̂) |
|---|---:|---:|---:|
| 0.01 | 0.005 | 0.005 | 0.0039 |
| 0.05 | 0.050 | 0.049 | 0.0039 |
| 0.10 | 0.101 | 0.096 | 0.0039 |

- 拟合律（独立 2000 批经验频率）下校准的真律超越率与标称相容 → L2 在均匀弱误差律下失配代价小；TV 粗界（T5 空泛风险）按原样保留为开放问题，不据此声称覆盖真实训练依赖（等级 4/5）。

## S7/A3-S8 L3 整次运行校准（真实预测器机制矩阵：LP / prototype / GCN5f crossfit）

| 数据 | 预测器(机制) | arm | α | 评估池超越率 [95% CI] | 真实训练次数 | 秒 |
|---|---|---|---|---|---:|---:|
| cora-cc | LP(标签集随机) | run-pool | 0.01 | 0.010 [0.000,0.054] | 250 | 2.5 |
| cora-cc | LP(标签集随机) | run-pool | 0.05 | 0.040 [0.011,0.099] | 250 | 2.5 |
| cora-cc | LP(标签集随机) | run-pool | 0.10 | 0.080 [0.035,0.152] | 250 | 2.5 |
| cora-cc | LP(标签集随机) | node-fit | 0.01 | 0.000 [0.000,0.036] | 250 | 2.5 |
| cora-cc | LP(标签集随机) | node-fit | 0.05 | 0.000 [0.000,0.036] | 250 | 2.5 |
| cora-cc | LP(标签集随机) | node-fit | 0.10 | 0.000 [0.000,0.036] | 250 | 2.5 |
| cora-cc | LP(标签集随机) | edge-iid | 0.01 | 0.000 [0.000,0.036] | 250 | 2.5 |
| cora-cc | LP(标签集随机) | edge-iid | 0.05 | 0.000 [0.000,0.036] | 250 | 2.5 |
| cora-cc | LP(标签集随机) | edge-iid | 0.10 | 0.000 [0.000,0.036] | 250 | 2.5 |
| dblp-ca | LP(标签集随机) | run-pool | 0.01 | 0.000 [0.000,0.045] | 200 | 116.6 |
| dblp-ca | LP(标签集随机) | run-pool | 0.05 | 0.062 [0.021,0.140] | 200 | 116.6 |
| dblp-ca | LP(标签集随机) | run-pool | 0.10 | 0.113 [0.053,0.203] | 200 | 116.6 |
| dblp-ca | LP(标签集随机) | node-fit | 0.01 | 0.000 [0.000,0.045] | 200 | 116.6 |
| dblp-ca | LP(标签集随机) | node-fit | 0.05 | 0.000 [0.000,0.045] | 200 | 116.6 |
| dblp-ca | LP(标签集随机) | node-fit | 0.10 | 0.000 [0.000,0.045] | 200 | 116.6 |
| dblp-ca | LP(标签集随机) | edge-iid | 0.01 | 0.000 [0.000,0.045] | 200 | 116.6 |
| dblp-ca | LP(标签集随机) | edge-iid | 0.05 | 0.000 [0.000,0.045] | 200 | 116.6 |
| dblp-ca | LP(标签集随机) | edge-iid | 0.10 | 0.000 [0.000,0.045] | 200 | 116.6 |
| cora-cc | proto(半split随机) | run-pool | 0.01 | 0.020 [0.004,0.057] | 280 | 12.1 |
| cora-cc | proto(半split随机) | run-pool | 0.05 | 0.033 [0.011,0.076] | 280 | 12.1 |
| cora-cc | proto(半split随机) | run-pool | 0.10 | 0.087 [0.047,0.144] | 280 | 12.1 |
| cora-cc | proto(半split随机) | node-fit | 0.01 | 0.000 [0.000,0.024] | 280 | 12.1 |
| cora-cc | proto(半split随机) | node-fit | 0.05 | 0.000 [0.000,0.024] | 280 | 12.1 |
| cora-cc | proto(半split随机) | node-fit | 0.10 | 0.000 [0.000,0.024] | 280 | 12.1 |
| cora-cc | proto(半split随机) | edge-iid | 0.01 | 0.033 [0.011,0.076] | 280 | 12.1 |
| cora-cc | proto(半split随机) | edge-iid | 0.05 | 0.093 [0.052,0.152] | 280 | 12.1 |
| cora-cc | proto(半split随机) | edge-iid | 0.10 | 0.093 [0.052,0.152] | 280 | 12.1 |
| dblp-ca | proto(半split随机) | run-pool | 0.01 | 0.000 [0.000,0.045] | 150 | 80.6 |
| dblp-ca | proto(半split随机) | run-pool | 0.05 | 0.062 [0.021,0.140] | 150 | 80.6 |
| dblp-ca | proto(半split随机) | run-pool | 0.10 | 0.125 [0.062,0.218] | 150 | 80.6 |
| dblp-ca | proto(半split随机) | node-fit | 0.01 | 0.000 [0.000,0.045] | 150 | 80.6 |
| dblp-ca | proto(半split随机) | node-fit | 0.05 | 0.000 [0.000,0.045] | 150 | 80.6 |
| dblp-ca | proto(半split随机) | node-fit | 0.10 | 0.000 [0.000,0.045] | 150 | 80.6 |
| dblp-ca | proto(半split随机) | edge-iid | 0.01 | 0.000 [0.000,0.045] | 150 | 80.6 |
| dblp-ca | proto(半split随机) | edge-iid | 0.05 | 0.000 [0.000,0.045] | 150 | 80.6 |
| dblp-ca | proto(半split随机) | edge-iid | 0.10 | 0.000 [0.000,0.045] | 150 | 80.6 |
| cora-cc | gcn5f(5折crossfit) | run-pool | 0.01 | 0.000 [0.000,0.060] | 120 | 2770.6 |
| cora-cc | gcn5f(5折crossfit) | run-pool | 0.05 | 0.067 [0.018,0.162] | 120 | 2770.6 |
| cora-cc | gcn5f(5折crossfit) | run-pool | 0.10 | 0.100 [0.038,0.205] | 120 | 2770.6 |
| cora-cc | gcn5f(5折crossfit) | node-fit | 0.01 | 0.000 [0.000,0.060] | 120 | 2770.6 |
| cora-cc | gcn5f(5折crossfit) | node-fit | 0.05 | 0.000 [0.000,0.060] | 120 | 2770.6 |
| cora-cc | gcn5f(5折crossfit) | node-fit | 0.10 | 0.000 [0.000,0.060] | 120 | 2770.6 |
| cora-cc | gcn5f(5折crossfit) | edge-iid | 0.01 | 0.000 [0.000,0.060] | 120 | 2770.6 |
| cora-cc | gcn5f(5折crossfit) | edge-iid | 0.05 | 0.000 [0.000,0.060] | 120 | 2770.6 |
| cora-cc | gcn5f(5折crossfit) | edge-iid | 0.10 | 0.133 [0.059,0.246] | 120 | 2770.6 |

- 已测 run-pool 的 α=5%/10% 点估计及条件区间见表；这是有限样本读数，不主张精确命中目标。个别 α=1% 单元校准预算不足 q=∞，该设置无检出能力，单独标注。
- node-fit 在**所有**真实运行机制下零告警（保守未报，非误报）（包括 GCN 5-fold crossfit——run 间预测差异 10.5%）：乘积律重采样的分数分布系统性**高于**真实运行分数分布，阈值过保守、0 告警。方向与计划 §3.4 预警一致（失配是保守/零告警方向，不是误报方向）；L1 下 edge-iid 的过度超越与 L3 下 node-fit 的保守零告警**不可混用为同一叙事**。
- **Gate B3 判定**：整次运行分位数作为本研究 L3 的主要实际基线；node 校准的价值在层级保证 + 失配诊断 + 影响结构解释（§8.2 power 读数），不是 run-pool 的省钱替代。成本证据：LP 250 训练 2.5 秒；GCN 120 训练 46 分钟——预算权衡故事成立但结论方向不变。DBLP GCN 运行池成本外推 4–12 小时，移入批量窗口（followups F3），不阻塞判定。

## S9a 归一化抵消的线性响应分解（评审改造一）

在独立参考中心 p 处 S(a)-S0 = J_A(E)+resid，J = 关系项 R1 + 度抵消项 Rdeg（C = D0^{-1}diag(E1)）；两臂在同一冻结基点上测（共同固定中心口径）。

| 结构 | ratio_R1 | ratio_J | ratio_true | S2 ratio_S | 抵消node | 抵消edge | resid/true node |
|---|---:|---:|---:|---:|---:|---:|---:|
| familyA_coherent/fixed | 1.08 | 0.45 | 1.29 | 0.24 | 0.53 | -0.12 | 1.15 |
| familyB_flat/gamma=0.05_cancel | 1.01 | 1.01 | 1.00 | 1.00 | -0.07 | -0.07 | 0.91 |
| familyB_flat/gamma=0.95_max | 0.95 | 0.53 | 1.18 | 0.53 | 0.40 | -0.07 | 1.03 |
| cora-cc/binary-eps0.05 | 1.38 | 1.39 | 1.28 | 1.24 | -0.15 | -0.14 | 1.15 |
| dblp-ca/binary-eps0.05 | 1.77 | 1.59 | 1.20 | 1.13 | -0.01 | -0.12 | 1.85 |

- **辅助导数通道分析**：ratio_R1（固定度加权后的关系项）已把 coherent 的 raw 2.93 压到 ~1.1——第一层抵消来自静态 D0^{-1/2} 加权（S2 中 ratio_S0=1.16 与此一致）；动态度修正项 Rdeg 是第二层，对相干块再压（canc_frac 0.4-0.5，familyA ratio_J≈0.45），对原生图净抵消弱甚至轻微增强（canc_frac≈0）——这些读数显示局部导数通道随结构改变，不单独解释S2波动比反转。
- **判定（降级版，按评审第三轮）**：静态度加权显著改变放大程度；线性响应揭示了度修正的抵消通道；但有限幅度随机扰动仍受显著高阶项支配——familyA 上 ratio_J=0.45 与同基点 ratio_true=1.29 分处 1 的两侧，方向并不一致，resid/true>1。当前数据不能写成“已解释真实归一化后的反转”；历史 S10a 范数差下界也不能认证有效范围。
- 注：表中 cos 列为 Frobenius 内积除以谱范数乘积的粗指标（可超 ±1），负值表示关系项与度抵消项方向相反（抵消）。

## S9b 方向标准化检测 T_g = max_j |v_j^T(S-M_hat)v_j|/(sigma_j+eps)（评审改造二）

inf/var/rand组在参考阶段冻结，目标alpha=5%；实际正常误报率见normal行，不假定每个冻结阈值都精确达到5%。本表是方向生成修复前的历史探索。

| 数据 | 机制 | norm | inf | var | rand |
|---|---|---:|---:|---:|---:|
| cora-cc | M1_x1.5 | 0.090 | 0.078 | 0.080 | 0.755 |
| cora-cc | M1_x2.0 | 0.117 | 0.080 | 0.095 | 0.997 |
| cora-cc | M1_x3.0 | 0.175 | 0.128 | 0.118 | 1.000 |
| cora-cc | M2_hubs_x4 | 0.225 | 0.165 | 0.162 | 1.000 |
| cora-cc | M2_lowdeg_x4 | 0.057 | 0.048 | 0.068 | 0.063 |
| cora-cc | normal | 0.048 | 0.058 | 0.052 | 0.058 |
| dblp-ca | M1_x1.5 | 0.064 | 0.108 | 0.036 | 1.000 |
| dblp-ca | M1_x2.0 | 0.092 | 0.136 | 0.084 | 1.000 |
| dblp-ca | M1_x3.0 | 0.108 | 0.236 | 0.088 | 1.000 |
| dblp-ca | M2_hubs_x4 | 0.136 | 0.268 | 0.120 | 1.000 |
| dblp-ca | M2_lowdeg_x4 | 0.060 | 0.064 | 0.044 | 1.000 |
| dblp-ca | normal | 0.062 | 0.052 | 0.050 | 0.078 |

- **cora-cc 描述**：漂移机制 5 格中 inf>norm 有 0 格。只报告点估计次序；未做配对差异确认，不把图间差异归因为hub集中度。
- **dblp-ca 描述**：漂移机制 5 格中 inf>norm 有 5 格。只报告点估计次序；未做配对差异确认，不把图间差异归因为hub集中度。
- 未建立方向统计的稳定优势。rand为有效竞争读数，均值与波动都可能进入响应；hub集中度解释仅为假设。方向生成修复后需要重新冻结并复跑，S9c也不提供选择方向优越性的普遍保证。

## S9c 匹配边缘依赖拨盘（逐边边缘 p_e 严格不变、只改依赖结构；动态归一化后均值与波动可同变，非纯波动机制）

**基线 cal_A_x1.00**

| 数据 | x | norm | inf | var | rand | 漂移vsA |
|---|---|---:|---:|---:|---:|---:|
| cora-cc | x=1.00 | 0.049 | 0.057 | 0.057 | 0.077 | 0.038 |
| cora-cc | x=0.75 | 0.007 | 0.045 | 0.013 | 0.020 | 0.049 |
| cora-cc | x=0.50 | 0.000 | 0.047 | 0.000 | 0.028 | 0.070 |
| cora-cc | x=0.25 | 0.000 | 0.058 | 0.000 | 0.030 | 0.077 |
| cora-cc | x=0.00 | 0.000 | 0.045 | 0.000 | 0.035 | 0.088 |
| dblp-ca | x=1.00 | 0.058 | 0.050 | 0.042 | 0.058 | 0.097 |
| dblp-ca | x=0.75 | 0.000 | 0.000 | 0.044 | 0.520 | 0.135 |
| dblp-ca | x=0.50 | 0.000 | 0.000 | 0.020 | 0.948 | 0.147 |
| dblp-ca | x=0.25 | 0.000 | 0.000 | 0.000 | 1.000 | 0.162 |
| dblp-ca | x=0.00 | 0.000 | 0.000 | 0.008 | 0.984 | 0.165 |
| familyA_coherent | x=1.00 | 0.045 | 0.043 | 0.043 | 0.043 | 0.016 |
| familyA_coherent | x=0.75 | 0.052 | 0.042 | 0.770 | 0.002 | 0.324 |
| familyA_coherent | x=0.50 | 0.000 | 0.040 | 0.918 | 0.002 | 0.429 |
| familyA_coherent | x=0.25 | 0.000 | 0.000 | 0.990 | 0.000 | 0.505 |
| familyA_coherent | x=0.00 | 0.000 | 0.000 | 1.000 | 0.000 | 0.542 |
| familyB_flat | x=1.00 | 0.055 | 0.037 | 0.051 | 0.041 | 0.021 |
| familyB_flat | x=0.75 | 0.370 | 0.050 | 0.276 | 0.040 | 0.247 |
| familyB_flat | x=0.50 | 0.674 | 0.048 | 0.558 | 0.058 | 0.351 |
| familyB_flat | x=0.25 | 0.846 | 0.046 | 0.774 | 0.096 | 0.405 |
| familyB_flat | x=0.00 | 0.946 | 0.048 | 0.908 | 0.084 | 0.432 |

**基线 cal_B_x0.00**

| 数据 | x | norm | inf | var | rand | 漂移vsA |
|---|---|---:|---:|---:|---:|---:|
| cora-cc | x=1.00 | 0.998 | 0.310 | 0.228 | 0.083 | 0.036 |
| cora-cc | x=0.75 | 0.778 | 0.145 | 0.140 | 0.020 | 0.058 |
| cora-cc | x=0.50 | 0.315 | 0.102 | 0.118 | 0.028 | 0.069 |
| cora-cc | x=0.25 | 0.100 | 0.072 | 0.068 | 0.055 | 0.076 |
| cora-cc | x=0.00 | 0.037 | 0.062 | 0.053 | 0.045 | 0.084 |
| dblp-ca | x=1.00 | 1.000 | 0.364 | 0.192 | 0.000 | 0.105 |
| dblp-ca | x=0.75 | 0.992 | 0.244 | 0.144 | 0.000 | 0.128 |
| dblp-ca | x=0.50 | 0.764 | 0.088 | 0.068 | 0.008 | 0.147 |
| dblp-ca | x=0.25 | 0.204 | 0.056 | 0.060 | 0.044 | 0.160 |
| dblp-ca | x=0.00 | 0.044 | 0.056 | 0.092 | 0.048 | 0.161 |
| familyA_coherent | x=1.00 | 0.090 | 0.090 | 0.050 | 0.090 | 0.010 |
| familyA_coherent | x=0.75 | 0.128 | 0.128 | 0.060 | 0.128 | 0.321 |
| familyA_coherent | x=0.50 | 0.096 | 0.098 | 0.048 | 0.096 | 0.446 |
| familyA_coherent | x=0.25 | 0.110 | 0.118 | 0.082 | 0.114 | 0.512 |
| familyA_coherent | x=0.00 | 0.040 | 0.048 | 0.038 | 0.034 | 0.544 |
| familyB_flat | x=1.00 | 0.056 | 0.000 | 0.048 | 0.056 | 0.012 |
| familyB_flat | x=0.75 | 0.036 | 0.000 | 0.024 | 0.036 | 0.239 |
| familyB_flat | x=0.50 | 0.038 | 0.000 | 0.006 | 0.038 | 0.340 |
| familyB_flat | x=0.25 | 0.054 | 0.000 | 0.008 | 0.058 | 0.402 |
| familyB_flat | x=0.00 | 0.030 | 0.000 | 0.004 | 0.058 | 0.432 |

**基线 cal_mix_x0.50**

| 数据 | x | norm | inf | var | rand | 漂移vsA |
|---|---|---:|---:|---:|---:|---:|
| cora-cc | x=1.00 | 0.872 | 0.248 | 0.145 | 0.093 | 0.042 |
| cora-cc | x=0.75 | 0.287 | 0.107 | 0.045 | 0.035 | 0.059 |
| cora-cc | x=0.50 | 0.048 | 0.042 | 0.027 | 0.047 | 0.068 |
| cora-cc | x=0.25 | 0.013 | 0.042 | 0.018 | 0.055 | 0.074 |
| cora-cc | x=0.00 | 0.000 | 0.045 | 0.010 | 0.055 | 0.082 |
| dblp-ca | x=1.00 | 0.812 | 0.372 | 0.080 | 0.000 | 0.098 |
| dblp-ca | x=0.75 | 0.280 | 0.280 | 0.068 | 0.000 | 0.115 |
| dblp-ca | x=0.50 | 0.048 | 0.060 | 0.052 | 0.028 | 0.147 |
| dblp-ca | x=0.25 | 0.004 | 0.012 | 0.016 | 0.192 | 0.162 |
| dblp-ca | x=0.00 | 0.000 | 0.016 | 0.000 | 0.244 | 0.170 |
| familyA_coherent | x=1.00 | 0.102 | 0.102 | 0.046 | 0.102 | 0.025 |
| familyA_coherent | x=0.75 | 0.090 | 0.044 | 0.044 | 0.090 | 0.315 |
| familyA_coherent | x=0.50 | 0.038 | 0.028 | 0.028 | 0.036 | 0.433 |
| familyA_coherent | x=0.25 | 0.000 | 0.000 | 0.000 | 0.006 | 0.511 |
| familyA_coherent | x=0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.546 |
| familyB_flat | x=1.00 | 0.064 | 0.000 | 0.054 | 0.064 | 0.015 |
| familyB_flat | x=0.75 | 0.052 | 0.000 | 0.024 | 0.052 | 0.244 |
| familyB_flat | x=0.50 | 0.064 | 0.000 | 0.008 | 0.064 | 0.348 |
| familyB_flat | x=0.25 | 0.084 | 0.000 | 0.006 | 0.072 | 0.405 |
| familyB_flat | x=0.00 | 0.100 | 0.000 | 0.004 | 0.048 | 0.432 |

- **判定（按评审第三轮修正）**：inf 组响应高度依赖参考机制——cal_A 基线上平坦（不响应依赖丧失），但 cal_B/cal_mix 基线检测依赖出现时原生结构有明确响应（cora 24.8–31.0%、dblp 36.4–37.2%，远超名义 5%），同时刻 norm 为 87.2–100%——**inf 有响应但未稳定优于其他统计量；既未成为强方法，也未被普遍否定**。family 合成族上 inf≈0–10%（抵消结构方向无能量、符合理论预期），norm 同低。
- 均值成分说明：拨盘保持逐边边缘，但动态归一化下 E S_mix(x) ≠ E S_A——mean_drift_vs_A 列即证据（0.016→0.542）。共同冻结中心是合理监测协议，但不排除均值成分；纯波动研究需各分布自身中心化的对照（列入后续）。
## S10a 历史幅度网格 A(t)=A0+tE（共同基点响应与范数差下界）

| 结构 | 范数差代理跨 0.5 的网格 t | ratio 穿越 1 的网格 t | ratio(t=0.05) | ratio(t=0.5) | ratio(t=1) |
|---|---:|---:|---:|---:|---:|
| familyA_coherent/fixed | 0.2 | 1.0 | 0.53 | 0.50 | 1.29 |
| familyB_flat/gamma=0.95_max | 0.4 | 1.0 | 0.46 | 0.52 | 1.19 |
| cora-cc/binary-eps0.05 | 0.3 | nan | 1.39 | 1.16 | 1.27 |
| dblp-ca/binary-eps0.05 | 0.2 | nan | 1.50 | 1.25 | 1.19 |

- **度量修复**：历史 `resid_frac_node` 实际计算 E|‖ΔS‖₂−‖tJ‖₂| / E‖ΔS‖₂，而非矩阵剩余项 E‖ΔS−tJ‖₂ / E‖ΔS‖₂。反三角不等式只保证前者≤后者；代理跨 0.5 可以排除该点的良好一阶近似，代理低于 0.5 不能认证有效范围。撤回原 t≈0.2–0.4 的有效区间断言。
- 共同基点下比值的方向与各臂自身中心化波动（S2）不同，不称为同一反转的因果解释；网格穿越只定位到相邻采样幅度之间。历史 DBLP 中段曾有手工数值复核，未保存完整逐样本数值诊断；不能据此断言 S2/S3/S9 均未受影响。修复后的脚本将实际剩余项与求解残差写入 `amplitude_scan_v2.csv`，需另行重跑；本轮保留历史测量，S10a 不支撑主贡献。

## S10b 双侧校准检测 p_two = 2·min(p_up, p_lo)（同校准集秩、并列保守）

| 结构 | 机制 | stat | 单侧 | 双侧 |
|---|---|---|---:|---:|
| cora-cc | M1_x2 | inf | 0.073 | 0.077 |
| cora-cc | M1_x2 | norm | 0.133 | 0.113 |
| cora-cc | M1_x2 | rand | 0.997 | 0.980 |
| cora-cc | M1_x2 | var | 0.050 | 0.053 |
| cora-cc | M2_hubs | inf | 0.143 | 0.070 |
| cora-cc | M2_hubs | norm | 0.150 | 0.130 |
| cora-cc | M2_hubs | rand | 1.000 | 1.000 |
| cora-cc | M2_hubs | var | 0.083 | 0.047 |
| cora-cc | M2_lowdeg | inf | 0.023 | 0.013 |
| cora-cc | M2_lowdeg | norm | 0.040 | 0.043 |
| cora-cc | M2_lowdeg | rand | 0.063 | 0.050 |
| cora-cc | M2_lowdeg | var | 0.033 | 0.023 |
| cora-cc | dial_x0.00 | inf | 0.000 | 0.143 |
| cora-cc | dial_x0.00 | norm | 0.000 | 0.993 |
| cora-cc | dial_x0.00 | rand | 0.027 | 0.007 |
| cora-cc | dial_x0.00 | var | 0.010 | 0.057 |
| cora-cc | dial_x0.25 | inf | 0.000 | 0.140 |
| cora-cc | dial_x0.25 | norm | 0.000 | 0.987 |
| cora-cc | dial_x0.25 | rand | 0.017 | 0.000 |
| cora-cc | dial_x0.25 | var | 0.007 | 0.047 |
| cora-cc | dial_x0.50 | inf | 0.000 | 0.053 |
| cora-cc | dial_x0.50 | norm | 0.000 | 0.867 |
| cora-cc | dial_x0.50 | rand | 0.010 | 0.010 |
| cora-cc | dial_x0.50 | var | 0.010 | 0.030 |
| cora-cc | dial_x0.75 | inf | 0.000 | 0.060 |
| cora-cc | dial_x0.75 | norm | 0.003 | 0.537 |
| cora-cc | dial_x0.75 | rand | 0.017 | 0.027 |
| cora-cc | dial_x0.75 | var | 0.013 | 0.023 |
| cora-cc | normal_FPR_check | inf | 0.065 | 0.060 |
| cora-cc | normal_FPR_check | norm | 0.045 | 0.055 |
| cora-cc | normal_FPR_check | rand | 0.037 | 0.043 |
| cora-cc | normal_FPR_check | var | 0.055 | 0.045 |
| dblp-ca | M1_x2 | inf | 0.133 | 0.100 |
| dblp-ca | M1_x2 | norm | 0.093 | 0.040 |
| dblp-ca | M1_x2 | rand | 1.000 | 1.000 |
| dblp-ca | M1_x2 | var | 0.120 | 0.093 |
| dblp-ca | M2_hubs | inf | 0.267 | 0.127 |
| dblp-ca | M2_hubs | norm | 0.133 | 0.053 |
| dblp-ca | M2_hubs | rand | 1.000 | 1.000 |
| dblp-ca | M2_hubs | var | 0.180 | 0.087 |
| dblp-ca | M2_lowdeg | inf | 0.067 | 0.040 |
| dblp-ca | M2_lowdeg | norm | 0.053 | 0.053 |
| dblp-ca | M2_lowdeg | rand | 1.000 | 1.000 |
| dblp-ca | M2_lowdeg | var | 0.093 | 0.053 |
| dblp-ca | dial_x0.00 | inf | 0.000 | 0.280 |
| dblp-ca | dial_x0.00 | norm | 0.000 | 1.000 |
| dblp-ca | dial_x0.00 | rand | 0.993 | 0.993 |
| dblp-ca | dial_x0.00 | var | 0.013 | 0.120 |
| dblp-ca | dial_x0.25 | inf | 0.000 | 0.193 |
| dblp-ca | dial_x0.25 | norm | 0.000 | 1.000 |
| dblp-ca | dial_x0.25 | rand | 0.993 | 0.987 |
| dblp-ca | dial_x0.25 | var | 0.007 | 0.100 |
| dblp-ca | dial_x0.50 | inf | 0.000 | 0.133 |
| dblp-ca | dial_x0.50 | norm | 0.000 | 0.887 |
| dblp-ca | dial_x0.50 | rand | 0.933 | 0.920 |
| dblp-ca | dial_x0.50 | var | 0.007 | 0.093 |
| dblp-ca | dial_x0.75 | inf | 0.000 | 0.053 |
| dblp-ca | dial_x0.75 | norm | 0.000 | 0.453 |
| dblp-ca | dial_x0.75 | rand | 0.540 | 0.467 |
| dblp-ca | dial_x0.75 | var | 0.047 | 0.047 |
| dblp-ca | normal_FPR_check | inf | 0.045 | 0.050 |
| dblp-ca | normal_FPR_check | norm | 0.040 | 0.050 |
| dblp-ca | normal_FPR_check | rand | 0.055 | 0.070 |
| dblp-ca | normal_FPR_check | var | 0.055 | 0.035 |
| familyA_coherent | dial_x0.00 | inf | 0.000 | 0.000 |
| familyA_coherent | dial_x0.00 | norm | 0.000 | 0.000 |
| familyA_coherent | dial_x0.00 | rand | 0.000 | 0.563 |
| familyA_coherent | dial_x0.00 | var | 0.000 | 0.000 |
| familyA_coherent | dial_x0.25 | inf | 0.000 | 0.000 |
| familyA_coherent | dial_x0.25 | norm | 0.000 | 0.000 |
| familyA_coherent | dial_x0.25 | rand | 0.000 | 0.690 |
| familyA_coherent | dial_x0.25 | var | 0.000 | 0.000 |
| familyA_coherent | dial_x0.50 | inf | 0.000 | 0.000 |
| familyA_coherent | dial_x0.50 | norm | 0.000 | 0.000 |
| familyA_coherent | dial_x0.50 | rand | 0.007 | 0.863 |
| familyA_coherent | dial_x0.50 | var | 0.000 | 0.000 |
| familyA_coherent | dial_x0.75 | inf | 0.000 | 0.000 |
| familyA_coherent | dial_x0.75 | norm | 0.000 | 0.000 |
| familyA_coherent | dial_x0.75 | rand | 0.000 | 0.837 |
| familyA_coherent | dial_x0.75 | var | 0.000 | 0.000 |
| familyA_coherent | normal_FPR_check | inf | 0.000 | 0.000 |
| familyA_coherent | normal_FPR_check | norm | 0.000 | 0.000 |
| familyA_coherent | normal_FPR_check | rand | 0.003 | 0.003 |
| familyA_coherent | normal_FPR_check | var | 0.000 | 0.000 |
| familyB_flat | dial_x0.00 | inf | 0.000 | 0.000 |
| familyB_flat | dial_x0.00 | norm | 0.000 | 0.000 |
| familyB_flat | dial_x0.00 | rand | 0.000 | 0.523 |
| familyB_flat | dial_x0.00 | var | 0.047 | 0.000 |
| familyB_flat | dial_x0.25 | inf | 0.000 | 0.000 |
| familyB_flat | dial_x0.25 | norm | 0.000 | 0.000 |
| familyB_flat | dial_x0.25 | rand | 0.000 | 0.553 |
| familyB_flat | dial_x0.25 | var | 0.063 | 0.000 |
| familyB_flat | dial_x0.50 | inf | 0.000 | 0.000 |
| familyB_flat | dial_x0.50 | norm | 0.013 | 0.003 |
| familyB_flat | dial_x0.50 | rand | 0.000 | 0.497 |
| familyB_flat | dial_x0.50 | var | 0.037 | 0.000 |
| familyB_flat | dial_x0.75 | inf | 0.000 | 0.000 |
| familyB_flat | dial_x0.75 | norm | 0.053 | 0.053 |
| familyB_flat | dial_x0.75 | rand | 0.000 | 0.297 |
| familyB_flat | dial_x0.75 | var | 0.050 | 0.000 |
| familyB_flat | normal_FPR_check | inf | 0.000 | 0.000 |
| familyB_flat | normal_FPR_check | norm | 0.037 | 0.025 |
| familyB_flat | normal_FPR_check | rand | 0.055 | 0.065 |
| familyB_flat | normal_FPR_check | var | 0.052 | 0.000 |

- 正常机制双侧 FPR 最大 0.070（标称 0.05）；仅为当前冻结阈值的经验读数，不能由最大点估计证明校准。双侧有效性来自两侧有效秩 p 值与并集界。
- **判读**：依赖丧失拨盘位点上 cora 的 norm “零告警”**主要是单侧方向伪影**（one=0.000 → two≈0.99），评审第五点被数据证实；但 familyA 上 norm 双侧仍为 0（one 与two 均 0.000）——其分数分布被 hub 翻转原子控制，B 臂分数落在校准 IQR 内，双侧化救不了原子遮蔽。两种机制（方向伪影 vs 原子遮蔽）必须分开陈述，检测能力评估需先看参考分数分布的形状。两点补充（DBLP）：漂移机制下非-rand 统计中 inf 最强（M2_hubs 26.7% vs norm 13.3%，与 S9b 一致），双侧轻微稀释（下尾无证据也占概率质量）；拨盘双侧化后 inf 也显出下尾响应（x=0 时 28.0%）——与评审“inf 有响应但不稳定占优”的表述一致。rand 对均值成分持续高响应（有效竞争结果，目标相关）。


## S11 固定训练划分、只改变初始化（L3 机制隔离）

半监督 GCN（固定 20% 标注、hidden64/epochs500(Cora)或120(DBLP)/lr0.2），run 随机性只来自初始化。

| 数据 | arm | α | 评估池超越率 [95% CI] | 训练次数 |
|---|---|---|---|---:|
| cora-cc | run-pool | 0.05 | 0.017 [0.000,0.089] | 120 |
| cora-cc | run-pool | 0.10 | 0.150 [0.071,0.266] | 120 |
| cora-cc | node-fit | 0.05 | 0.000 [0.000,0.060] | 120 |
| cora-cc | node-fit | 0.10 | 0.000 [0.000,0.060] | 120 |
| cora-cc | edge-iid | 0.05 | 0.000 [0.000,0.060] | 120 |
| cora-cc | edge-iid | 0.10 | 0.000 [0.000,0.060] | 120 |
| dblp-ca | run-pool | 0.05 | 0.025 [0.001,0.132] | 80 |
| dblp-ca | run-pool | 0.10 | 0.025 [0.001,0.132] | 80 |
| dblp-ca | node-fit | 0.05 | 0.000 [0.000,0.088] | 80 |
| dblp-ca | node-fit | 0.10 | 0.000 [0.000,0.088] | 80 |
| dblp-ca | edge-iid | 0.05 | 0.000 [0.000,0.088] | 80 |
| dblp-ca | edge-iid | 0.10 | 0.000 [0.000,0.088] | 80 |

- 判读：两个固定划分设置中，run-pool 条件区间均包含 α，但样本少（Cora 评估 60 次、DBLP 40 次），不构成等效性证明。Cora α=5% 1/60、α=10% 9/60；DBLP 两档 1/40。node-fit 均为 0 次超越；不能由 0 次观测推断总体超越率为零。区间宽度与评估样本量、条件超越概率有关，不能归因于分数分布窄。

## S12 新种子复核（修正区间与结论范围）

| 主张 | 结构 | 指标 | 估计 [95% CI] | 次数 | 区间方法 |
|---|---|---|---|---:|---|
| C1_S3_calibration_recovery | cora-cc | node_cal_exceedance_alpha05 | 0.037 [0.013,0.062] | 112/3000 | conditional_hoeffding |
| C3_S10b_two_sided_norm_dial0 | cora-cc | norm_power_one | 0.000 [0.000,0.009] | 0/400 | conditional_clopper_pearson |
| C3_S10b_two_sided_norm_dial0 | cora-cc | norm_power_two | 0.985 [0.968,0.994] | 394/400 | conditional_clopper_pearson |
| C2_S9b_M2hubs_detection | dblp-ca | M2_hubs_power_inf | 0.316 [0.259,0.378] | 79/250 | conditional_clopper_pearson |
| C2_S9b_M2hubs_detection | dblp-ca | M2_hubs_power_var | 0.004 [0.000,0.022] | 1/250 | conditional_clopper_pearson |
| C2_S9b_M2hubs_detection | dblp-ca | M2_hubs_power_rand | 1.000 [0.985,1.000] | 250/250 | conditional_clopper_pearson |
| C2_S9b_M2hubs_detection | dblp-ca | M2_hubs_power_norm | 0.240 [0.188,0.298] | 60/250 | conditional_clopper_pearson |

- C1：112/3000=3.73%；三批使用不同参考中心与阈值，原合并二项 CP 区间撤销。给定三套冻结工件后，独立有界评估的 Hoeffding 95% 区间为 [1.25%,6.21%]，对象是三批条件超越概率的样本量加权均值，不是校准重抽样的边际覆盖。不能写精确命中 5%，也不能从原区间判定秩保证失败。
- C2：DBLP inf 31.6%、norm 24.0%，为两组不同观测；缺少配对结果与本轮正常机制误报检查。仅保留描述性差异，不主张确认了检测优势。C3：双侧 394/400、单侧 0/400，上述区间已补齐；检验选择仍为辅助诊断。
- S12 未新增 L3 训练运行，不能作为真实预测器独立确认。脚本记录了另设种子，但未保存运行前带时间戳的冻结协议，因此称新种子复核，不追认预注册确认。

## 判定汇总

- **Gate A（主线继续）：可继续**（A1=True，A2=True，A4 见 S4 段）
- **Gate B（方法价值）：node 校准恢复目标（A2）+ 检出未崩溃（B2）= 初步支持继续。B3：run-pool 是已测 LP/proto/GCN 机制的主要实际基线，node-fit 均观察到零超越——论文定位为“层级保证 + 机制失配诊断 + 影响结构解释”，不主张 node 校准是 run-pool 的预算替代（详见 S7 段）。
- 后续围绕单一主线：按 claims_evidence.md 的三条主张整合稿件；若要将真实运行复核升级为独立确认，先保存前瞻协议、运行预测缓存和逐分数数值诊断，再用新种子检验。方向检测、新数据、T2b/T3 与 Gate D 均不作为本轮收尾的新增主贡献。