# AmbiHTMLBench

用于评测模型输出 HTML 对参考页面的还原能力，提供两条互补路径：

1. `JSON vs HTML`：用结构化 GT 评估内容、语义标签与布局，是可重复的客观测量。
2. `Reference + rendered HTML`：用 VLM Judge 近似正常人的整体视觉判断，弥补结构指标难以表达的可见重建质量。

客观路径保持不变；主观路径已升级为可预检、可恢复、双方向复核，并采用顺序无关的 Batch Bradley--Terry 聚合。

## 1. 环境安装

```bash
python -m pip install -e .
python -m playwright install chromium
```

运行测试时再安装开发依赖：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

入口命令：

```bash
python main.py -h
```

## 2. 路径1：GT JSON vs 预测 HTML

### 2.1 Ground Truth JSON 格式

推荐格式（坐标为相对坐标，范围 [0,1]）：

```json
{
  "width": 1280,
  "height": 1800,
  "elements": [
    {
      "id": "gt_1",
      "tag": "p",
      "text": "示例文本",
      "bbox": {"x": 0.1, "y": 0.2, "w": 0.7, "h": 0.06}
    }
  ]
}
```

兼容字段：
- `elements` / `items` / `nodes`
- `bbox` / `box`
- 坐标字段支持 `x,y,w,h` 或 `left,top,width,height` 或 `x1,y1,x2,y2`
- 若坐标是绝对像素，需提供 `width,height`，程序会自动归一化

上面的内联 JSON 即为最小示例。

额外兼容 `Quality_inspection/submitted_108_gt_final_submit.json` 这种聚合 GT 格式：
- 顶层是数组
- 每个页面对象包含 `ID`、`imageUrl`、`contentList`
- `contentList` 内主要字段：
  - `remark`：文本内容
  - `name` / `markUpEntityTagList[].tagName`：语义标签
  - `left` / `top` / `width` / `height`：绝对坐标
  - `points`：多边形点集（缺 bbox 时回退使用）

目前已内置标签归一化，例如：
- `标题title -> title`
- `正文文本text_block -> text_block`
- `列表类（目录）list / 合并列表类list_merge -> list`
- `图片figure -> figure`
- `表格table -> table`
- `行间公式equation_isolated -> equation_isolated`

### 2.2 先把 HTML 提取为结构化 JSON（调试用）

```bash
python main.py extract-html \
  --html path/to/pred.html \
  --out outputs/pred.json
```

### 2.3 单样本评测

```bash
python main.py eval-json-html \
  --gt-json path/to/gt.json \
  --html path/to/pred.html \
  --out outputs/eval_one.json
```

如果 `gt_json` 是 `Quality_inspection` 的聚合大文件，可以显式指定页 ID：

```bash
python main.py eval-json-html \
  --gt-json Quality_inspection/submitted_108_gt_final_submit.json \
  --gt-page-id 192 \
  --gt-image Quality_inspection/grok_4_5_infer_20260712_192743/downloaded_images/043_192_good_2605_18623v1.jpg \
  --html Quality_inspection/grok_4_5_infer_20260712_192743/rendered_html/043_192_good_2605_18623v1.html
```

也可以传 `--records-json`，程序会自动从 HTML 文件名推断 `page_id` 和对应原图：

```bash
python main.py eval-json-html \
  --gt-json Quality_inspection/submitted_108_gt_final_submit.json \
  --records-json Quality_inspection/grok_4_5_infer_20260712_192743/logs/records.json \
  --html Quality_inspection/grok_4_5_infer_20260712_192743/rendered_html/043_192_good_2605_18623v1.html
```

输出指标：
- `content_score`：匹配元素的文本相似度（归一化编辑距离）
- `tag_score`：匹配元素的语义标签相似度
- `page_iou`：页面级布局 IoU
- `f1`：元素匹配 F1
- `layout_score = 0.5 * page_iou + 0.5 * f1`
- `total_score = w_content * content_score + w_layout * layout_score`

默认 `w_content=0.5, w_layout=0.5`，可通过参数调整。

### 2.4 数据集批评测

方式A（推荐）：`pairs.jsonl`

每行：

```json
{"id":"sample_01","gt":"/abs/gt1.json","html":"/abs/pred1.html"}
```

运行：

```bash
python main.py eval-dataset-json-html \
  --pairs-jsonl pairs.jsonl \
  --out outputs/path1_scores.json
```

方式B：目录对齐（同 stem）

```bash
python main.py eval-dataset-json-html \
  --gt-dir path/to/gt_dir \
  --html-dir path/to/html_dir \
  --out outputs/path1_scores.csv
```

### 2.5 Quality Inspection 批量评测

直接评测你当前这套 `Quality_inspection` 数据：

```bash
python main.py eval-quality-inspection \
  --gt-json Quality_inspection/submitted_108_gt_final_submit.json \
  --records-json Quality_inspection/grok_4_5_infer_20260712_192743/logs/records.json \
  --out outputs/quality_inspection_scores.csv
```

这个命令会自动：
- 用 `records.json` 把 GT `ID`、原图、HTML 对上
- 用原图尺寸归一化 GT 坐标
- 用原图尺寸作为 HTML 渲染视口

## 3. 路径2：Reference-anchored 主观评测

### 3.1 为什么需要主观评测

结构化指标可以稳定度量文字、标签和几何关系，却不能完整回答“这个页面在正常人眼中是否像参考页、是否完整重建”这一整体问题。明显缺失的图表、表格或图片区块，严重的页面高度和布局失真，以及破坏阅读体验的文本区域错误，都应进入视觉判断；反之，肉眼几乎不可见的细小 OCR 字符差异不应主导结论。

因此，主观路径评估的是 **human-visible reconstruction fidelity to the Reference**，而不是扣除客观指标后的残差。七个 GT 引导轴只负责定位和诊断关键视觉对象，不构成固定加权公式：

1. 图片视觉保真度（`IMAGE_VISUAL_FIDELITY`）
2. 表格视觉结构（`TABLE_VISUAL_STRUCTURE`）
3. 图表视觉编码（`CHART_VISUAL_ENCODING`）
4. 公式二维布局（`FORMULA_2D_LAYOUT`）
5. 算法块结构（`ALGORITHM_BLOCK_STRUCTURE`）
6. 竖排文字呈现（`VERTICAL_TEXT_PRESENTATION`）
7. 列表层级呈现（`LIST_HIERARCHY_PRESENTATION`）

### 3.2 评测流程

主观评测分为三个显式阶段：

```text
public manifest
  -> arena-preflight：渲染、请求体积检查、候补页替换、平衡调度
  -> arena-run：Reference + A + B，canonical / swapped 双方向判决，断点续跑
  -> arena-report：方向归一化、pair collapse、Batch Bradley--Terry、bootstrap 与可靠性报告
```

每个物理模型对都以两种展示顺序各评一次：

- `canonical`：Reference + A + B
- `swapped`：Reference + B + A

归一化到相同物理模型顺序后：两次同向胜者保留胜者；`TIE / TIE` 记为感知平局（`perceptual`）；任何方向不一致记为可靠性平局（`reliability`）。可靠性平局表示 Judge 未能在换序后复现同一方向偏好，不表示两个候选在视觉上不可区分。

主排行榜对每个物理模型对只使用一个折叠结果，以未加权、顺序无关的 Batch Bradley--Terry 最大似然模型联合拟合，评分中心为 1000、尺度为 400。置信度加权和未折叠双方向拟合只作为敏感性诊断。

### 3.3 使用方法

输入格式见 [`docs/MANIFEST.md`](docs/MANIFEST.md)；也可用 `scripts/build_manifest_from_dirs.py` 从目录生成 manifest。运行仓库内的示例前，请先把示例路径替换为实际 Reference、GT 和候选 HTML 路径。

先运行不产生 Judge API 费用的预检：

```bash
python main.py arena-preflight \
  --manifest examples/subjective_manifest.example.json \
  --out-dir outputs/preflight
```

仅将预检产出的 `resolved_manifest.json` 交给正式评测：

```bash
export VLM_API_BASE='https://your-provider.example/v1'
export VLM_API_KEY='...'
export VLM_JUDGE_MODEL='your-judge-model'

python main.py arena-run \
  --manifest outputs/preflight/resolved_manifest.json \
  --decisions-out outputs/arena/decisions.json \
  --concurrency 4
```

最后折叠双方向结果并生成排行榜、bootstrap 区间和可靠性诊断：

```bash
python main.py arena-report \
  --manifest outputs/preflight/resolved_manifest.json \
  --decisions-json outputs/arena/decisions.json \
  --out-dir outputs/arena/report \
  --bootstrap-replicates 1000
```

## 4. 目录结构

```text
benchmark/
  arena.py                 # 主观评测编排、预检与断点续跑
  arena_audit.py           # pair collapse、可靠性与 bootstrap
  arena_schedule.py        # 平衡 round-robin 调度
  rating.py                # Batch Bradley--Terry
  subjective.py            # 七轴、Prompt、解析与方向归一化
  prompts/subjective_v1.md # 冻结 Prompt；代码在导入时校验 SHA256
  render.py                # Chromium 复用与有界回退
  cli.py                   # 客观命令 + 主观三阶段命令
  html_extractor.py        # 原客观路径，保持不变
  schema.py
  scorer.py
  utils.py
configs/                   # 主观评测默认配置
docs/                      # 方法、架构、复现与结论边界
examples/                  # manifest 示例
scripts/                   # manifest 构建工具
tests/                     # unit / regression / integration / golden replay
main.py
pyproject.toml
requirements.txt
requirements-dev.txt
```

## 5. 验证结论与表述边界

系统级方案已通过冻结 Final Arena90 重放、Fresh45 端到端留出验证、历史顺序依赖回归、双方向一致性检查和盲审面板。验证支持主观评测规范、可靠性协议、聚合算法和工程链路的改进。

盲审结果**不能证明**新版 Prompt 的人类一致性准确率高于旧 Prompt，也不能把 VLM 判决等同于人类判决。新版 Prompt 被保留，是因为它落实了修正后的评测目标，并与双方向可靠性和 Batch Bradley--Terry 系统完整集成。详见 [`docs/FINAL_SYSTEM_VALIDATION_REPORT.md`](docs/FINAL_SYSTEM_VALIDATION_REPORT.md) 与 [`docs/VALIDATION_CLAIMS.md`](docs/VALIDATION_CLAIMS.md)。
