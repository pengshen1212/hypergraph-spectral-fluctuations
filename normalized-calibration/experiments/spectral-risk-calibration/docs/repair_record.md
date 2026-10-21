# 第一篇后续修复记录（2026-10-21）

本轮范围：修复第一篇的实验代码、证据推断、主张映射与状态说明，保持单一主线：
**归一化影响分析 → 指定随机律下校准 → 真实运行失配诊断**。
验证范围是单元测试和保存结果的确定性重新分析，未重新训练或复现全部科学实验。

## 修复内容

| 问题 | 修复 | 历史证据处理 |
|---|---|---|
| S12-C1三批不同阈值却合并为同一二项样本 | 独立有界观测的条件Hoeffding区间 | 112/3000原计数不变；新95%区间[1.25%,6.21%]；旧CP列不再引用 |
| S12-C3未给区间 | 从整数计数重建条件CP区间 | 0/400为[0,0.92%]，394/400为[96.76%,99.45%] |
| C2方向点估计被当作已确认优势 | 明确非配对、缺少新FPR检查；保守差值区间跨0 | inf79/250 vs norm60/250保留，差值7.6pp仅作描述 |
| 秩边际保证与固定阈值条件率混用 | 理论笔记补概率对象和证明；修复报告、等级与表注 | 不由点估计等于/低于α作保证判断，不由区间包含α宣称等效 |
| S9a/S10a共同基点被当作各臂自身中心化 | S2自身均值、S9a/S10a的S(p)、监测M_hat三者分开 | 不混成一条反转因果解释 |
| S10a用两个范数之差冒充矩阵剩余项 | v2计算‖ΔS−tJ‖，同时保留范数差代理和求解残差 | 旧CSV不变；撤回有效区间断言；v2全量未跑 |
| 随机方向除以另一随机向量的范数，eigsh未显式固定起点 | 使用自身范数并固定起点 | 历史方向结果仍为修复前探索；将来写入v2文件，不混用 |
| T4应用说明过度限制引理且混淆最小度与κ | 一般正度引理不变，d0≥κ，节点替换保留deg(u)因子 | T4不作为新尾证书；ES点估计仍不合格证书 |
| 主张清单使探索分析与主证据并列 | M1/M2/M3组织全文；方向/幅度/尾界移辅助 | 见claims_evidence.md与manuscript_integration.md |

## 修改入口

- `scripts/audit_evidence.py` → `results/evidence_audit.json`：可重建计数、有效区间、S11条件区间及原数据哈希。
- `scripts/build_gate_report.py` → `docs/gate_ab_report.md`：源头修复，避免重建时恢复旧断言。
- `scripts/s9b_directional_detection.py`、`s9c_dependence_dial.py`、`s10a_amplitude_scan.py`、`s10b_two_sided.py`、`s12_new_seed_confirm.py`：受修复影响的未来科学结果独立写入v2。
- `tests/test_evidence_repairs.py`：8项新增回归，包含异质批次、缺失区间、矩阵剩余项、方向生成与报告解释。
- README、claims/status/followups/operator/theorem notes统一；新增稿件整合规格。

## 本轮实际验证

1. 原18项单元测试通过；新增回归先观察到对应失败，再修复。最终`python -m pytest tests -q`：**26 passed**。
2. `python -m compileall -q scripts src tests`：退出码0。
3. audit与gate report连续生成两次，输出SHA-256相同。
4. 对照修复前哈希：**19个既有CSV/JSON文件全部不变**。备份在`../archive/20261021-before-followup-repair/`。

## 尚未完成的研究与稿件工作

不把本轮测试称为实验独立复现：未重跑S9/S10/S12全量，未新增L3训练；未证明DBLP历史其他阶段不受数值异常影响。若需要这些更强主张，按followups.md先冻结协议、保留逐样本缓存与诊断，再运行。

`paper/paper1/main.tex`、原投稿包、第二篇和既有release包本轮未修改。正文整合与文献创新性核验仍需后续完成；已有具体整合规格，不默认增加新检测器、降权或下游任务。
