# 本轮新增引用核验（2026-10-21）

本轮仅新增两条直接支持校准定位的文献，未把原稿14条参考文献全部重新检索。

| 引用键 | 原始来源核验 | 本文使用范围 |
|---|---|---|
| vovk2012conditional | [PMLR原始出版页](https://proceedings.mlr.press/v25/vovk12.html)：Vladimir Vovk，Conditional Validity of Inductive Conformal Predictors，ACML，PMLR 25，475–490，2012 | 说明边际有效性与不同条件有效性对象的区别；本文另给所用简单秩规则的完整证明 |
| angelopoulos2024crc | [作者机构出版页](https://research.google/pubs/conformal-risk-control/)与[作者预印本页](https://arxiv.org/abs/2208.02814)：Angelopoulos、Bates、Fisch、Lei、Schuster，Conformal Risk Control，ICLR 2024 | 定位既有单调损失期望风险控制；不将本文算子尾报警称为新CRC理论，也不据此声称分类风险控制 |

正文没有借用新文献证明本实验的效果或创新性，未声称完成全面最近邻排重。
BibTeX保留原14条条目；校准文献2条及下述ED-HNN条目加入后共17条。ICLR和AllSet会议条目没有填造页码。
当前elsarticle-num样式给出3条非致命书目字段提示：旧AAAI条目同时有volume/number，AllSet与CRC条目无pages。LaTeX引用解析与PDF编译另行核验。

2026-10-22：新增 `wang2023edhnn`，用于正文§5.4、数据可用性和附录的视觉数据预处理来源归属。已核对[官方实现及作者BibTeX](https://github.com/Graph-COM/ED-HNN)和[作者预印本](https://arxiv.org/abs/2207.06680)：Peihao Wang、Shenghao Yang、Yunyu Liu、Zhangyang Wang、Pan Li，Equivariant Hypergraph Diffusion Neural Operators，ICLR 2023。此引用不用于支持本实验的校准结论。ICLR条目无页码会产生一条非致命提示。
