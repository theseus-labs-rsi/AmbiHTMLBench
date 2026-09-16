# AmbiHTMLBench 三裁判委员会选拔证据报告

## 结论摘要

- 原 v1 冻结委员会：`anthropic_sonnet5,gemini31pro,openai_terra`。
- v1 在 320 题 Pilot 选拔集上按预先固定的字典序规则排名第 1：弃权感知 Brier=0.1258，覆盖率=77.81%。
- 独立的 1,552 题 holdout 验证没有复现 v1 的第一名；v1 排名第 4，Brier=0.1293，覆盖率=78.99%。
- holdout 第一名为 `anthropic_opus5,gemini31pro,openai_sol`：Brier=0.1240，覆盖率=80.35%，加权二元准确率=85.48%。
- v1 减 v2 的 holdout 配对 ΔBrier=+0.0053，95% CI [-0.0001, +0.0106]；负值才支持 v1。页面整块 Bootstrap 中 v1 优于 v2 的概率仅 2.64%。
- v2 在 holdout 的委员会整块 Bootstrap 中获得第一名的概率为 76.81%，平均名次 1.27。
- 但稀有特殊轴并未完整进入 holdout：公式、算法块、列表和竖排轴均被结构化抽样集中在 Pilot。对全量中样本数≥30的公式/表格/竖排三条特殊轴做等权宏平均时，v1 Brier=0.1268，优于 v2 的 0.1324。

**审计结论：** v1 的 Pilot 选拔过程有效且无泄漏，但“它是全量人类判断下总体最接近人的三人委员会”这一强结论未通过独立验证。负责人已于 2026-08-30 明确将“按 1,872 题自然频率最小化总体人类偏差”定为最终目标，因此冻结 v2 `anthropic_opus5,gemini31pro,openai_sol`。历史 v1 冻结文件保留不变，v2 另存为 `final_frozen_committee_config.json`。这一决策必须与特殊轴上 v1 的描述性优势及 v2 的选拔后偏差限制一并披露。

## 最终冻结决策

- 目标：总体人类对齐（overall human alignment）。
- 最终委员会：`anthropic_opus5,gemini31pro,openai_sol`。
- 版本：`committee-selection-v2-overall-human-alignment`。
- 历史 v1 仅作为 Pilot 时点的预定规则冻结记录，不做删除或回写。
- 选拔后限制：v2 是查看 holdout 后确立的最终方案，所以 holdout 是选拔证据，不是对 v2 的全新独立验证集。

## 可复核的方法

1. 人类目标为 `p(A)=(A票+0.5×TIE票)/有效偏好票`，按有效人数/5加权。
2. 每个候选裁判先对 `2方向×2重复` 的四个规范化标签做 `3-of-4` 折叠，不足则输出 Reliability Tie。
3. 三裁判对 A/B/感知 TIE 做 `2-of-3` 多数决，其余依协议记为 Reliability Tie 或 Disagreement。
4. 主指标为弃权感知、人类票数加权 Brier；未解析输出以中性 `p(A)=0.5` 计入，防止通过大量弃权虚高准确率。
5. Pilot 320 题只用于记录 v1 当时的选拔；其余 1,552 题用于验证。
6. 在每个题卷分层内对整页重抽样 10,000 次，保留同页多个模型对的相关性。
7. 主比较保留三家族各一席与覆盖率≥75%；所有 10 种三模型组合亦完整披露。

## Pilot 选拔与 Holdout 验证

| 分割 | 委员会 | 合格组合排名 | 覆盖率 | 弃权感知 Brier | 条件 Brier | 加权二元准确率 |
|---|---|---:|---:|---:|---:|---:|
| Pilot | v1: anthropic_sonnet5,gemini31pro,openai_terra | 1 | 77.81% | 0.1258 | 0.1279 | 86.02% |
| Pilot | v2: anthropic_opus5,gemini31pro,openai_sol | 4 | 79.06% | 0.1404 | 0.1438 | 82.78% |
| Holdout | v1: anthropic_sonnet5,gemini31pro,openai_terra | 4 | 78.99% | 0.1293 | 0.1366 | 84.18% |
| Holdout | v2: anthropic_opus5,gemini31pro,openai_sol | 1 | 80.35% | 0.1240 | 0.1301 | 85.48% |

## 必须披露的限制

- v2 是根据 holdout 结果提出的最终部署建议，因而 holdout 对 v2 不再是一个完全未见的二次验证集。证据包额外给出留一题卷选择稳定性和整页 Bootstrap，但不把它们包装成新的独立实验。
- holdout 不是特殊轴完备的验证集。它含固定轴和表格轴，却不含公式、算法块、列表与竖排轴；这些轴只能用 Pilot/全量切片作描述性比较。
- Gemini 是唯一的 Google 家族候选者，所有跨家族委员会都包含它。它的自家族模型偏好必须在 `self_preference_audit.csv` 中单独披露。
- 不应宣称某组合“显著优于所有其他组合”；应同时报告效应量、区间、覆盖率和选择稳定性。

## 文件索引

- `committee_candidates_by_split.csv`：10 种委员会在 Pilot/Holdout/全量的完整结果。
- `single_judges_by_split.csv`：5 位候选裁判的人类对齐和稳定性。
- `holdout_pairwise_bootstrap.csv`：v1、v2 与其他合格组合的配对不确定性。
- `holdout_selection_stability.csv`：候选委员会的 Bootstrap 获胜概率和平均名次。
- `committee_slices.csv`：v1/v2 的题卷、轴和敏感性切片。
- `special_axis_balance.csv`：四个合格委员会在全部特殊轴上的并列比较及宏平均。
- `self_preference_audit.csv`：裁判对自家族/同模型候选页的残差偏好审计。
- `leave_one_paper_out.csv`：留一题卷的委员会选择稳定性。
- `committee_evidence.json`：机器可读的完整摘要和数据指纹。
