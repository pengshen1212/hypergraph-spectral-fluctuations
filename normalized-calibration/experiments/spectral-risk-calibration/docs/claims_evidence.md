# 主张—证据映射（跨域校准叙事已修订，2026-10-22）

当前稿：`paper/paper1/main.tex`与`main.pdf`，26页。M1对应§3/§5.1及表1，M2对应§4/§5.2及表2、表3，M3对应§5.3及表4；视觉跨域校准在§5.4、表5、图2及附录H.2；附录H.3仅保留次要特征扰动响应。原raw理论与旧实验在附录A/B，支持性归一化扰动工具在附录H。完整块保留与数值来源核验见`paper/paper1/notes/calibration-integration/verification.json`。

唯一研究问题：**节点诱导依赖如何改变归一化谱风险，以及校准需要保留哪一层随机律？**

论证顺序：归一化影响分析 → 指定随机律下的秩校准 → 真实运行的机制失配诊断。
这是同一问题的解释、保证和适用边界，不拆成三个算法贡献。条件期望分解与秩规则是既有工具，不宣称新的一般概率理论。

## 三条正文主张

| 主张 | 公式与依据 | 实验与结果 | 稿件安排 | 证据边界 |
|---|---|---|---|---|
| M1 归一化改变共享节点影响与波动，不能直接继承原始算子的放大/抵消结论 | F_u^S(z)=E[S∣Y_u=z]−ES；X=ΣF_u^S+R；能量正交；T1见theorem_notes.md | s1/s2/s4；exact_checks.json、normalization_pilot.csv、influence_residual.csv | 归一化影响节；三算子表与同度影响对照 | A/B匹配EB，不自动匹配ES；S2各臂自身中心化；native影响仅为节点子集 |
| M2 校准分数须保留假设的节点诱导关系依赖；在指定节点律内有有限样本秩保证 | T=‖S−M_hat‖₂；k=ceil((M+1)(1−α))；T_new>q；给定独立参考阶段后对校准与新观测联合取概率 | s3/s6与Cora补充重跑；calibration_baselines.csv、l2_fitted_law.csv、cora_calibration_batches_20271021.csv/.json | 校准节与模型内结果表 | L1模型保证不迁移到真实训练；L2为经验失配评估；新Cora三批均值5.03%，批次区间端点包络2.28%–8.65%；未前瞻注册，非精确命中证明 |
| M3 真实训练的完整运行律不由节点边缘乘积律决定；run-pool是实际校准基线 | 同一冻结中心，比较完整预测向量抽样与节点频率乘积重采样 | s7/s8/s11；l3_run_pool.csv、l3_fixed_split.csv | 真实运行节；机制分组的超越率表 | 只覆盖已测协议；S11两个固定训练划分；0次超越不等于总体概率为0；S12没有新增L3确认 |

## 逐项审计与位置

T6：①已证明解析命题；②指定节点模型内秩保证；③同一运行机制下交换性秩保证；
④数值/经验结果（包括已知律模拟与拟合律实验）；⑤跨机制失配压力测试。
解析依据和数值证据分开；数值通过不升级为①，非校准恒等式不标②。

| ID | 可使用的主张 | 公式—脚本—结果 | 等级 | 主线地位及限制 |
|---|---|---|---|---|
| A1/A6 | 独立节点坐标下归一化条件影响分解与能量恒等式精确成立 | T1证明；s1_exact_checks.py→exact_checks.json；s4→influence_residual_summary.json | ①；数值④ | M1；小图精确核验；受控族MC为支持，native子集不宣称全量闭合 |
| A2 | 固定度线性映射保留共同均值，动态度一般不保留 | S0(a)=D0^(−1/2)A(a)D0^(−1/2)；s1/s2 | ①；数值④ | M1；均值移位单列 |
| A3 | 已测原生单元动态归一化后的波动比仍大于1，放大量小于raw | ratio_S=E_A‖S−E_A S‖₂ / E_B‖S−E_B S‖₂；s2→normalization_pilot.csv | ④ | M1；12单元，不外推所有结构；相干族比值<1及均值移位分别报告 |
| A4 | 指定节点律下节点校准有秩保证；独立边阈值在该律下可过度超越 | 秩阈值；s3→calibration_baselines.csv | ②；数值④⑤ | M2；PASS是内部筛选，端点包络不是合并95%CI |
| A5 | 二分类影响能量P0P1‖Δ_u‖_F²；同度hub影响可不同 | Δ_u=E[S∣1]−E[S∣0]；s4→influence_residual.csv | 恒等式①；数值④ | M1；不是已证的最优检测方向 |
| A7 | T4扰动引理在正度下成立，hub替换保留度因子 | theorem_notes.md；s5_theory_numeric.py→theory_checks.json | ①；数值④ | 附录；不是非空泛尾风险主界 |
| A8 | ES重要性估计重尾，点估计不能充当证书 | s5/s5b→theory_checks.json、es_ucb.json | ④ | 局限；T2b/T3不作本轮主贡献 |
| A9/A10 | 已测完整运行校准与节点乘积重采样的读数不同，固定划分初始化机制亦如此 | s7/s8/s11→l3_run_pool.csv、l3_fixed_split.csv | run-pool③；数值④⑤ | M3；不称run-pool“唯一有效”；S11区间宽因评估量小，不归因于分数窄 |
| A11 | 共同基点导数分解显示度修正通道 | J=D0^(−1/2)ED0^(−1/2)−(CS(p)+S(p)C)/2；s9a→normalization_response.csv | ④ | 辅助；S(p)不是ES，不能与S2自身中心化视为同一量 |
| A12 | 历史共同基点比值随幅度改变；旧剩余项列仅为范数差代理 | s10a→amplitude_scan.csv；未来v2→amplitude_scan_v2.csv | ④ | 附录；撤回“一阶有效区间t≈0.2–0.4”；v2未全量重跑，DBLP数值排查未完成 |
| A13 | 方向统计依赖机制与参考选择，无稳定优势证据 | s9b/s10b→directional_detection.csv、two_sided_detection.csv | 校准依据②；数值④ | 辅助；C2为非配对描述；未确认hub集中度因果解释；方向生成修复后须重跑才有新结果 |
| A14 | Cora依赖丧失拨盘的双侧norm响应高于单侧；familyA仍可不响应 | s10b/s12-C3→two_sided_detection.csv、confirm_results.csv | 校准依据②；数值④ | 辅助；C3 0/400 vs 394/400，条件CP区间见audit |
| A15 | 拨盘保留逐边边缘，但归一化均值与波动可同变 | mix_flags；s9c→dependence_dial.csv | 构造①；数值④ | 辅助；不是纯波动识别 |
| A16 | 新种子复核提供限定的模型内校准与双侧响应读数 | s12→confirm_results.csv；audit_evidence.py→evidence_audit.json | ②解析背景；数值④ | C1主线复核；C2/C3辅助；未保存前瞻冻结工件，不称预注册确认，不标③ |
| A17 | 视觉对象超图上L2节点乘积仍保守；L3在99次校准下并非一致名义；edge失配方向随数据变化 | s13/s14/s15→visual_alarm_v1/；protocol_frozen.json、e1_calibration_*.csv | 数值④⑤ | M3外部边界；资源版单批30/99/200；NTU L3 14.0%，ModelNet 5.5%；L2两数据0%；不得把14%改阈值或加跑至5% |
| A18 | 视觉特征噪声下任务事件少且报警精确率低；固定回退不提供操作收益 | s15→e3/e4 CSVs；s13b→development/lambda_inertia_*.json（审计材料） | 数值④ | 2026-10-22起退出论文叙事，原实验计划及负结果继续随包保存；不作分类或工程收益主张 |
| A19 | 门控细化对任意λ∈[0,1]保持预测类别；五策略相同是结构限制，不是训练不足 | 历史稿命题H.2及证明（20261022-before-cross-domain-focus存档）；monitoring_workflow.py refine_probs；s13b诊断 | ①解析；数值④ | 2026-10-22起从论文删除，不另立本篇贡献；历史证明和诊断继续保留，后续行动需独立协议 |
| A20 | NTU 14%是冻结阈值的一次实现；近似并列只解释0.5pp；重复拆分显示上尾不稳定；事后PAC规则为9.5% | s15b→analysis/threshold_stability_*.json | 数值④ | 内部拆分敏感性，不是新增独立运行；PAC为看过旧评价后的事后分析，不称已修复 |
| A21 | 按实际误报率比较，T在σ=1时仍不优于Frobenius/熵 | s15c→equal_fpr_roc.csv、equal_fpr_bootstrap.json、ROC图 | 数值④ | 同样本回顾性ROC，阈值非部署规则；不称严格支配或已校准检测器 |

## S12推断修复

- C1三批分别拟合中心/阈值，不是一个共同p的二项样本。原合并CP区间撤销；条件于三套冻结工件，独立Bernoulli均值的Hoeffding 95%区间为[1.25%,6.21%]。对象是样本量加权条件率，不是重抽校准集的边际覆盖。
- C2 inf与norm使用不同评估观测，缺少配对差异与本轮正常FPR检查。差值7.6pp的保守非配对Hoeffding区间跨0，不主张确认优势。
- C3补齐CP区间：单侧0/400：[0,0.92%]；双侧394/400：[96.76%,99.45%]。对象为冻结工件下的条件率。
- 历史CSV不改，必须通过audit或修复报告解释，不再引用C1旧CP列。

## 正文措辞约束

1. 秩保证是上侧控制，不保证超越率等于α，也不自动对某次实现的阈值条件成立。
2. run-pool是已测真实机制的实际基线，节点乘积重采样不是其低成本替代。
3. 报警表示算子偏离参考机制，不能直接解释为分类错误或准确率下降。
4. 方向检测、幅度扫描、更强尾界与风险降权不进入摘要贡献清单。
5. 视觉L3在NTU的14%条件误报必须如实报告；不能称“校准通过”，也不能在看过结果后改阈值、换种子或加跑到5%。
6. λ=0使五种应用策略全同；不得把惰性回退写成工程收益或任务纠错能力。
7. 保类结论依赖非负、同类连接和正对角；不得改写成“预测非常自信所以不变”，也不是待由更多训练确认的效应。
8. 等FPR ROC是同样本回顾性分数比较，不称部署阈值或独立确认；不得用原始名义阈值下的检出率直接宣称支配。

复现：`python scripts/audit_evidence.py`、`python scripts/build_gate_report.py`、`python -m pytest tests -q`。
稿件安排见`manuscript_integration.md`，当前状态见`status.md`，修复记录见`repair_record.md`。
