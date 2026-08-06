# Elo 机制逐版本开发记录

本文件用于记录 `feature/elo-arena` 分支中 Elo 机制的每一版设计、代码变化、测试结果和后续方向。

## 开发原则

- 所有版本都在同一个分支中持续开发。
- 每一版使用独立 Git commit 保存。
- 每次修改后先进行语法检查和自动测试。
- 新版本不能删除旧版本的对照能力。
- 稳定性提升不等同于真实排名精度提升，最终仍需结合真实数据验证。

## V0：原始 Confidence-Weighted Online Elo

### 状态

仓库原始基线版本。

### 主要机制

- 所有模型从相同初始分数开始。
- 按照 decisions 文件中的存储顺序逐场更新。
- 胜、负、平分别记为 1.0、0.0、0.5。
- 使用标准 400 分尺度的 Elo 预期得分公式。
- 使用裁判 confidence 调整单场 K 值。
- confidence 权重为 `0.5 + 0.125 * confidence`。

### 已知问题

- 最终评分会受到比赛记录排列顺序影响。
- benchmark 样本之间没有真实时间顺序，但原始在线 Elo 将其视为连续发生的比赛。
- 相同比赛集合采用不同排列时，可能得到不同 Elo 分数甚至不同排名。

## V1：Order-Robust Shuffle-Ensemble Elo

### 状态

已完成代码实现和基础自动测试。

### 修改文件

- `benchmark/arena.py`
- `benchmark/cli.py`
- `tests/test_arena_elo.py`

### 核心设计

1. 保留原始 `online` 模式，用于基线对照。
2. 新增默认的 `shuffle_ensemble` 模式。
3. 在随机排列前，先对比赛记录建立确定性标准顺序。
4. 使用固定随机种子生成多种比赛排列。
5. 每一种随机排列同时计算正序和倒序。
6. 对全部运行得到的模型 Elo 分数取平均。
7. 保留 V0 的 confidence 加权 K 值。
8. 增加参数合法性检查和异常提示。

### 新增命令行参数

- `--elo-mode`：选择 `online` 或 `shuffle_ensemble`。
- `--shuffle-repeats`：设置随机排列次数。
- `--elo-seed`：设置可复现的随机种子。

### V1 主要目标

- 降低 decisions 原始存储顺序造成的偶然偏差。
- 保证相同比赛集合和相同随机种子得到可复现结果。
- 保留标准 Elo 的核心更新公式和可解释性。
- 保留旧模式，便于后续实验比较。

### 已通过测试

- 空比赛记录返回空排名。
- 同分模型平局后维持初始分数。
- 高 confidence 产生更大的更新幅度。
- 获胜更多的模型获得更高评分。
- 原始 online Elo 确实具有顺序敏感性。
- shuffle ensemble 不依赖输入列表的初始排列。
- 固定随机种子时结果可复现。
- 所有模型的 Elo 总分保持守恒。

### 当前限制

- V1 主要改善顺序稳定性，尚不能单独证明真实排名精度提高。
- 同一样本、同一模型对的多轮裁判结果仍被视为多场独立比赛。
- confidence 仍直接采用裁判模型自报值，尚未进行可靠性校准。
- 当前只返回平均 Elo，没有输出不同排列下的标准差或置信区间。

## 后续候选方向

以下方向尚未实现，将逐版选择和验证：

- V2：已完成，多轮裁判结果聚合后再执行 Elo。
- V3：已完成，输出 Elo 均值、标准差和排名稳定性指标。
- V4：已完成，按每个模型在样本中的对局暴露量归一化 Elo 更新。
- V5：已完成，新增不依赖比赛顺序的全局同步 Batch Elo。
- V6：尝试 Bradley-Terry 全局能力拟合，与在线 Elo 对照。

## 版本提交规则

每完成一版，按以下顺序执行：

1. 运行语法检查。
2. 运行全部自动测试。
3. 检查 Git diff。
4. 更新本文件。
5. 创建该版本的独立 commit。
6. 推送到 `feature/elo-arena`。
7. 再开始下一版。

## V2：Multi-Round Evidence Aggregated Elo

### 状态

已完成代码实现和自动测试。

### 修改文件

- `benchmark/arena.py`
- `tests/test_arena_elo.py`
- `docs/elo_versions.md`

### 动机

V1 解决了比赛顺序敏感问题，但仍存在一个 benchmark 特有的问题：

同一个 sample、同一对模型在多个 round 中进行裁判时，原始 Elo 会将每一次判断视为独立比赛。

例如：

- round 1：A 胜 B
- round 2：A 胜 B
- round 3：A 胜 B

如果直接累计，会导致同一个样本因为重复评测而被过度放大。

### 核心设计

V2 将每一次裁判结果转换为 Elo evidence，而不是直接作为独立比赛。

新增：

- `EloEvidence` 数据结构；
- 多轮结果聚合函数；
- 一致性计算机制。

### 聚合策略

对于相同：

- `sample_id`
- `model_a`
- `model_b`

的多轮结果：

1. 将胜负结果转换为连续得分：
   - A 胜：1.0
   - 平局：0.5
   - B 胜：0.0

2. 使用 confidence 作为基础权重。

3. 根据多轮结果方差计算一致性：

- 多轮判断一致：
  - 提高可信度；
  - 增加更新影响。

- 多轮判断冲突：
  - 降低可信度；
  - 减少更新影响。

4. 引入递减轮数奖励：

多轮一致不会线性增加影响。

例如：

- 3 次一致判断不会等价于 3 场独立比赛；
- 但会比单次判断具有更高可信度。

### 新增参数

- `round_aggregation`
  - `none`：
    - 保留传统逐轮处理。
  - `sample_pair`：
    - 聚合同一样本同一模型对的多轮判断。

- `consistency_floor`
  - 控制冲突裁判结果最低可靠程度。

### V2 主要目标

- 避免同一样本多轮评测导致 Elo 被重复放大。
- 利用多轮信息提升可信度。
- 利用裁判一致性降低不确定样本影响。
- 保持单轮情况下与 V1 兼容。

### 已通过测试

在 V1 的 8 项测试基础上新增：

- 单轮聚合与无聚合结果一致。
- 多轮一致不会线性累积影响。
- 多轮冲突会降低更新幅度。
- 开启和关闭 round aggregation 会产生不同结果。

总测试：

- 12 / 12 passed。

### 当前限制

- 当前 sample 难度仍未参与 Elo 更新。
- 不同参考图片的信息量默认相同。
- confidence 仍来自裁判模型自报告，没有额外校准。
- 尚未输出 Elo 稳定性指标。

## V3：Uncertainty-Aware Elo

### 状态

已完成代码实现、自动测试和 CLI 端到端验证。

### 动机

V1 和 V2 提高了评分稳定性，但最终结果仍只包含一个平均 Elo。单一数值无法表达模型排名对比赛排列顺序的敏感程度，也无法说明模型的领先是否稳定。

### 核心设计

V3 将 shuffle ensemble 的每一次正序或倒序计算保留为一个独立 trial，再根据全部 trial 计算排名不确定性。

新增 `EloStats`，包含：

- `elo`：所有 trial 的平均 Elo。
- `elo_std`：不同 trial 下 Elo 的总体标准差。
- `mean_rank`：模型在所有 trial 中的平均名次。
- `rank_stability`：模型最常出现名次的出现比例。
- `top_rank_probability`：模型在 trial 中并列或独占最高 Elo 的比例。
- `trial_count`：参与统计的 trial 数量。

### 代码重构

新增统一的 `_run_elo_trials`：

- `compute_elo` 使用各 trial 的平均评分；
- `compute_elo_stats` 使用相同 trial 计算不确定性；
- 避免复制 shuffle ensemble 实现；
- 保证普通 Elo 与统计 Elo 使用完全相同的比赛排列。

新增并列名次处理：

- Elo 差异在容差内时视为并列；
- 并列模型使用平均名次；
- top rank probability 允许多个模型同时处于第一组。

### CLI 改进

新增：

- `--include-stats`
- `--round-aggregation`
- `--consistency-floor`

使用 `--include-stats` 时，输出平均 Elo、标准差、平均名次、排名稳定度和第一名概率。

### 已通过测试

在 V2 的 12 项测试基础上新增：

- 统计平均 Elo 与 `compute_elo` 完全一致。
- 单次 online trial 的 Elo 标准差为零。
- 所有统计指标处于合法范围。
- 循环胜负关系产生非零 Elo 不确定性。
- 明确的传递性胜负关系保持稳定第一名。

总测试：

- 17 / 17 passed。

### 当前限制

- 当前不确定性主要反映比赛排列顺序，而不是数据集采样误差。
- 尚未使用 bootstrap 对 sample 进行重采样。
- 排名稳定度不是统计显著性检验。
- 样本难度和信息量仍未参与 Elo 更新。

## V4：Sample-Balanced Elo

### 状态

已完成代码实现、自动测试和 CLI 验证。

### 动机

同一张参考样本会让所有有效模型两两比较。当参与模型数量增加时，产生的对局数量按组合数增长。

例如：

- 2 个模型产生 1 场比较；
- 3 个模型产生 3 场比较；
- 4 个模型产生 6 场比较。

如果每场比较都使用完整 Elo 更新权重，参与模型更多的样本会天然获得更大的总影响。模型输出缺失数量不一致时，这种影响尤其明显。

### 核心设计

新增 sample exposure normalization。

对于每个 sample，统计每个模型参与的证据数量，也就是模型在该样本对局图中的 degree。

每条 A 与 B 的证据使用以下缩放：

`scale = 2 / (degree_A + degree_B)`

在完整的 m 模型循环赛中，双方 degree 都为 m - 1，因此：

`scale = 1 / (m - 1)`

这样每个模型在一张样本中的总 Elo 暴露量约为 1，而不会随参与模型数量线性增加。

### 新增参数

- `sample_balance`
  - `none`：保留 V3 之前的逐证据完整权重。
  - `per_model_exposure`：按每个模型在样本中的对局暴露量进行归一化。

CLI 新增：

- `--sample-balance`

默认值为：

- `per_model_exposure`

### 兼容性

- 只有两个模型的样本缩放系数为 1，因此行为不变。
- Elo 更新仍保持零和，总评分守恒。
- V1 的顺序集成、V2 的多轮聚合和 V3 的不确定性统计均继续生效。
- 可以使用 `sample_balance=none` 完整关闭 V4，进行消融对照。

### 已通过测试

在 V3 的 17 项测试基础上新增：

- 两模型样本在启用归一化后保持不变。
- 多模型密集样本的总 Elo 波动被降低。
- 样本归一化后 Elo 总分仍然守恒。
- 非法 sample balance 模式会抛出明确异常。

总测试：

- 21 / 21 passed。

### 当前限制

- 当前归一化只考虑参与对局数量，没有直接估计样本真实难度。
- 不完整对局图使用双方 degree 的对称缩放，是一种启发式近似。
- 尚未验证该归一化对人工排名一致性的实际提升。
- 尚未进行 sample bootstrap 数据集重采样。

## V5：Global Batch Elo

### 状态

已完成代码实现、自动测试、数值审计和 CLI 验证。

### 动机

V1 通过多次随机排列和正反序平均降低在线 Elo 的顺序偏差，但底层仍然按照比赛顺序逐条更新。

V5 新增全局同步 Batch Elo。每次迭代使用当前评分同时计算全部 evidence 的误差，再同步更新所有模型，从机制上消除 evidence 排列顺序的影响。

### 核心设计

每轮迭代执行：

1. 根据当前 Elo 计算所有 evidence 的预期得分。
2. 计算加权误差 `actual - expected`。
3. 为每个模型累积全部 evidence 的梯度。
4. 按模型总 exposure 归一化梯度。
5. 所有模型同步更新，而不是逐场更新。
6. 将评分重新居中，使平均 Elo 保持为初始评分。
7. 达到收敛阈值或最大迭代次数后停止。

### 正则化

新增 L2 shrinkage，使模型评分向初始 Elo 轻微收缩。

作用：

- 防止全胜或全负模型的评分无限扩大。
- 在数据量较少时降低极端评分。
- 为断开的模型比较图提供稳定锚点。

### 新增模式

- `online`：保留原始逐场在线 Elo。
- `shuffle_ensemble`：保留 V1 的多顺序平均 Elo。
- `batch`：新增全局同步迭代 Elo。

### 新增参数

- `batch_iterations`：最大同步迭代次数。
- `batch_tolerance`：收敛阈值。
- `batch_regularization`：向初始 Elo 收缩的正则强度。

CLI 新增：

- `--batch-iterations`
- `--batch-tolerance`
- `--batch-regularization`

### 与已有版本的组合

Batch Elo 继续使用：

- V2 的多轮 evidence 聚合。
- V4 的 sample exposure normalization。
- confidence 和裁判一致性权重。

V3 的统计接口同样支持 Batch 模式。由于 Batch 模式只有一个确定性的全局 trial，因此其排列顺序标准差为零；这不代表不存在数据采样不确定性。

### 已通过测试

在 V4 的 21 项测试基础上新增：

- Batch Elo 与输入 evidence 顺序无关。
- 明确的传递性胜负关系得到正确排名。
- 对称循环胜负保持评分平衡。
- 平均 Elo 保持为初始评分。
- 更强正则化会降低评分跨度。
- Batch 统计模式只有一个确定性 trial。
- 非法迭代、阈值和正则参数会抛出异常。

总测试：

- 28 / 28 passed。

### 当前限制

- 当前算法是同步梯度式 Batch Elo，不是严格的 Bradley-Terry 最大似然求解器。
- 默认学习步长仍由 `k_factor` 控制。
- Batch 模式的零标准差只表示没有排列顺序波动。
- 数据集采样误差仍需要 sample bootstrap 估计。
- 尚未用人工偏好排名验证 Batch 模式的实际相关性提升。
