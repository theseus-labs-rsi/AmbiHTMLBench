# AmbiHTMLBench

AmbiHTMLBench 用于评测模型把页面图像重建为可执行 HTML 的能力。仓库目前包含两条互补评测路径：

1. **客观评测**：渲染模型输出的 HTML，抽取可见页面 block，并与人工 GT 做页面级匹配和打分。
2. **主观评测**：以 Reference 页面为锚点，让 VLM Judge 进行成对视觉偏好判断，并用 Batch Bradley--Terry 聚合。

客观评测当前主入口是 `eval-objective-v3`。旧版 `eval-json-html`、`eval-dataset-json-html`、`eval-quality-inspection` 等接口已不再是 README 推荐路径。

## 1. 环境安装

```bash
python -m pip install -e .
python -m playwright install chromium
```

运行测试时安装开发依赖：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

入口命令：

```bash
python main.py -h
```

## 2. 客观评测

### 2.1 输入文件组织

默认 GT 文件：

```text
AmbiHTMLBench_GT_v5_1115.json
```

该文件是固定 schema 的页面列表。每个页面包含 `ID`、`imageUrl` 和 `contentList`。`contentList` 中的 block 使用 `name` 表示类型，`remark` 表示文本、表格 HTML 或公式内容，`left/top/width/height` 表示 GT 框。

默认模型输出目录：

```text
all_model_infer/
  model_a/
    html/
      0001_xxx.html
      0002_xxx.html
  model_b/
    html/
      0001_xxx.html
```

模型目录名会作为 leaderboard 中的 `model` 名称。每个 HTML 文件名前四位是页面 ID，例如 `0006_xxx.html` 对应 GT 中的 `ID=6`。

### 2.2 运行全量客观评测

```powershell
python main.py eval-objective-v3 `
  --gt-json AmbiHTMLBench_GT_v5_1115.json `
  --infer-dir all_model_infer `
  --out-dir outputs/objective_v3
```

Linux/macOS shell 可以把行尾的反引号替换为反斜杠，或者写成单行命令。

常用调试参数：

```powershell
python main.py eval-objective-v3 `
  --models gpt-5.6-sol,gemini-3.1-pro-preview `
  --page-ids 0006,0359 `
  --out-dir outputs/objective_debug
```

```powershell
python main.py eval-objective-v3 `
  --max-pages 20 `
  --no-visualizations `
  --out-dir outputs/objective_smoke
```

输出文件：

- `page_scores.csv`：每个模型、每个页面的页面级指标。
- `leaderboard.csv`：每个模型的聚合分数和排序。
- `page_scores.partial.csv` / `leaderboard.partial.csv`：断点中间结果。
- `visualizations/<model>/<page_id>.html`：表格、文本、layout graph 匹配可视化。
- `visualizations/<model>/_assets/<page_id>.png`：模型 HTML 渲染截图。

### 2.3 页面元素抽取

预测 HTML 会先在 headless Chromium 中渲染，再抽取可见 DOM block。抽取逻辑包括：

- 可见文本节点会被合并到语义 block。
- 表格保留为完整 table block，不会在主抽取阶段拆成单元格。
- `img/svg/canvas/video/figure/table/math/.MathJax/.katex` 等结构节点会作为结构 block 抽取。
- `class` 包含 `placeholder` 的元素会被排除，不参与匹配和打分。
- 不可见文本、透明文本、极小字号文本、`display:none` 和 `visibility:hidden` 内容会被排除。

### 2.4 匹配逻辑

客观评测不是直接比较 HTML 源码，而是在 block 层面比较 GT 和预测结果。

**表格匹配**

- GT 表格和预测表格先做一对一匹配。
- 候选分数综合表格 TEDS、TEDS-S、文本相似度和几何重叠。
- 使用 Hungarian assignment 求全局一对一匹配。
- 匹配成功的表格用于计算 TEDS 和 TEDS-S。
- 未匹配到的 GT/预测表格会被拆成单元格文本，放入后续文本池。

**公式处理**

- 当前版本不再单独计算公式 CDM。
- GT 和预测中的公式会去掉 `$` 包裹后作为普通文本加入文本池。
- 因此 `formula_cdm` 在当前 leaderboard 中通常为空。

**文本匹配**

- 文本池包含普通文本、公式文本和未匹配表格拆出的单元格文本。
- 先基于 `partial_ratio` 文本相似度建立 GT-text 到 Pred-text 的二分图。
- 连通分量支持一对一、一对多、多对一、多对多。
- 分量内部会尝试阅读顺序、横向顺序、反向顺序、图引导顺序和小规模局部重排。
- 固定顺序后使用分段 DP 得到多对多文本组。
- 页面级 `text_edit_distance` 最终基于匹配组拼接后的 GT 文本和预测文本计算。
- 文本编辑距离用字符级 Levenshtein，归一化分母为 `max(len(gt), len(pred), 1)`。
- GT 文本中的 `ξ` 被视作通配符，可匹配任意单字符。

### 2.5 指标

每个页面输出以下核心指标：

- `table_teds`：表格内容与结构相似度，使用行/单元格序列编辑的归一化 TEDS 近似实现，范围 `[0,1]`，越高越好。只在 GT 有表格的页面计算。
- `table_teds_s`：只考虑表格结构、不考虑单元格文本的 TEDS-S 近似实现，范围 `[0,1]`，越高越好。只在 GT 有表格的页面计算。
- `formula_cdm`：当前版本不单独计算公式指标，通常为空。
- `text_edit_distance`：整页文本编辑距离，范围通常为 `[0,1]`，越低越好。
- `layout_graph_similarity`：二维 layout graph 相似度，范围 `[0,1]`，越高越好。

模型聚合时：

- `table_teds` 和 `table_teds_s` 只在有 GT 表格的页面上平均。
- `text_edit_distance` 和 `layout_graph_similarity` 在全部页面上平均。
- weighted score 使用：

```text
text_quality = 1 - text_edit_distance
weighted_score = (text_quality + layout_graph_similarity + 0.3 * table_teds) / 2.3
```

其中 TEDS-S 是诊断指标，不额外进入 weighted score，避免重复计算表格结构。

### 2.6 Layout Graph 分数

layout 只基于成功匹配的对象计算：

- 一个匹配表格贡献一个节点。
- 一个文本匹配组贡献一个节点。
- 一对多或多对多文本组会先取 GT 侧和预测侧各自的包络框。
- GT 或预测任一侧为空的文本组不进入 layout graph。

每个节点比较中心点位移和宽高相似度。然后在 GT 节点中心上构建 `k=3` 的 KNN 图，并用相同节点身份读取预测侧对应边，比较边的方向、长度和二维位移。最终分数由边相似度、节点相似度和匹配覆盖率加权得到：

```text
layout_graph_similarity = 0.55 * edge_score + 0.30 * node_score + 0.15 * coverage_f1
```

### 2.7 可视化

默认会为每个页面生成可视化 HTML。可视化页面包含：

- GT 原图和 GT 匹配框。
- 预测 HTML 渲染截图和预测匹配框。
- 表格匹配、文本匹配和 layout graph 边。
- 每个匹配组对应的 GT 文本和预测文本。

这部分主要用于人工检查匹配是否合理。

## 3. 主观评测

### 3.1 当前论文协议：冻结 OLS5 代理

主观评测比较 Reference 与两个候选页面的视觉还原质量。全局布局、文本呈现是固定诊断轴；表格、公式、算法、竖排文本、列表层级按参考结构激活。轴分数用于解释差异，不进入总体代理。

人工校准集包含 1,872 个物理比较。五位裁判各做双方向、两次重复，共 37,440 个判断；每裁判四个 canonical overall 判断的均值作为 OLS 特征。冻结后，部署改为每题每裁判单方向、单次：10,000 个比较、50,000 个判断。五个 overall 判断按固定系数组合，得到连续偏好分数，再拟合 soft Bradley–Terry 排名；区间来自 2,000 次整页 bootstrap。

原始单次部署与候选更新版分别保留：GPT-5.6-Sol 的 BT 分数为 154.34 和 154.70。当前论文使用后者。更新接受 54 个 model-page 候选，影响 98 个比较，其中 21 个属于旧人工子集；旧标签不能验证这些更新后的候选。全 10,000 场均由代理评分，不能声称获得了全量人工 Accuracy/Recall/F1。

### 3.2 离线验证与复现

新入口 `benchmark.frozen_proxy_cli` **不联网、不调用裁判 API**；仅使用该离线层不需要安装或启动浏览器，也不需要 API key。本次同步包含代码、冻结配置、轻量证据与两版汇总榜单，不包含完整原始判断、图片或人工标注。以下 `artifacts/` 路径是用户另行提供的外置数据，不是仓库已附文件。

```bash
python -m benchmark.frozen_proxy_cli validate \
  --results artifacts/candidate_refresh/single_results.jsonl \
  --config configs/subjective_ols5_frozen.json \
  --artifact-manifest evidence/subjective/dataset_versions.json \
  --dataset candidate_refresh

python -m benchmark.frozen_proxy_cli replay \
  --results artifacts/candidate_refresh/single_results.jsonl \
  --config configs/subjective_ols5_frozen.json \
  --artifact-manifest evidence/subjective/dataset_versions.json \
  --dataset candidate_refresh \
  --out-dir outputs/candidate_refresh_replay \
  --bootstrap 2000
```

输出目录必须是新目录，不覆盖冻结结果。省略 artifact manifest 时只能验证输入结构，不能认证为论文发布数据；`--fixture` 仅用于小型测试，不是论文复现。校准数据的离线 OLS/留一题卷复现见 [冻结代理说明](docs/FROZEN_PROXY.md)。

本次迁移没有提供新的图片→渲染→API 全流程。仓库现有 `all_model_infer/` 仍是原始候选集，并未包含上述 54 个更新；当前客观 GT v5 也不能替代冻结主观实验的旧快照。详见 [版本与来源说明](docs/SUBJECTIVE_SYNC_PROVENANCE.md)。

### 3.3 历史 Arena v1

`arena-preflight`、`arena-run`、`arena-report` 及旧配置继续保留，用于历史单裁判双方向实验；其中 `arena-run` 会调用付费 API。方向不一致折为 reliability TIE、1000 中心 BT、round-robin block bootstrap 等旧语义不属于当前 OLS5 部署。历史方法和验证边界见 [Method v1](docs/METHOD.md) 与 [旧验证记录](docs/REPRODUCIBILITY.md)，旧测试和冻结 prompt 均不改写。

## 4. 目录结构

```text
benchmark/
  objective_v3.py          # 当前客观评测：匹配、指标、leaderboard、可视化
  html_extractor.py        # Chromium 渲染后的 block 抽取
  cli.py                   # 客观 eval-objective-v3 + 历史 Arena v1 命令
  frozen_proxy_cli.py      # 当前主观代理：离线 validate / replay / calibrate
  schema.py                # 页面元素数据结构
  scorer.py                # 旧版基础 scorer，非当前主客观榜单入口
  arena.py                 # 历史 v1：编排、预检与断点续跑
  arena_audit.py           # 历史 v1：pair collapse、可靠性与 block bootstrap
  arena_schedule.py        # 历史 v1：平衡 round-robin 调度
  rating.py                # 历史 v1：离散结果的 Batch Bradley--Terry
  subjective.py            # 历史 v1：七轴、Prompt、解析与方向归一化
  render.py                # Chromium 复用与有界回退
scripts/
  convert_json_infer_to_model_dirs.py
  rebuild_objective_v5_reports.py
  merge_lowpage_rerun_results.py
  update_paper_objective_eval.py
tests/
  unit/test_objective_v3.py
configs/
  subjective_ols5_frozen.json
evidence/subjective/       # 脱敏轻量证据、两版榜单和来源哈希；不含原始判断全集
main.py
pyproject.toml
requirements.txt
requirements-dev.txt
```

## 5. 注意事项

- `outputs/` 和 `all_model_infer/` 通常很大，默认不建议提交。
- `AmbiHTMLBench_GT_v5_1115.json` 是当前客观评测默认 GT 文件。
- 若只想快速检查代码是否可运行，建议使用 `--max-pages` 或 `--page-ids`。
- 若只需要 CSV 分数而不需要人工检查匹配，可加 `--no-visualizations` 降低运行时间和输出体积。
