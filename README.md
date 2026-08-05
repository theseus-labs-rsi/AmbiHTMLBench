# HTML Reconstruction Benchmark

用于评测「模型输出 HTML 对原页面的还原能力」。提供两条评测路径：

1. `JSON vs HTML`：对比 ground truth JSON 和预测 HTML 的结构/内容差异。
2. `Image + HTML (VLM Arena)`：将 HTML 渲染为图像，通过 VLM 竞技场打分并用 Elo 排名模型。

## 1. 环境安装

```bash
python -m pip install -r requirements.txt
python -m playwright install chromium
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

完整示例见 [ground_truth_schema_example.json](./ground_truth_schema_example.json)。

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

## 3. 路径2：Image + HTML 的 VLM 竞技场 Elo

### 3.1 机制

- 对每个样本，把每个模型的 HTML 渲染成图片。
- 对同一样本里的模型两两对战（A/B）。
- 输入给 VLM：`参考原图 + 候选A图 + 候选B图`。
- VLM 输出：`winner: A/B/TIE`、`confidence: 1..5`。
- 聚合全部对战结果，计算 Elo 排名。

### 3.2 判决 Prompt 设计

内置评审维度权重：
- 版面结构与几何布局 40%
- 文本内容正确性 35%
- 视觉样式 20%
- 关键元素保留 5%

防偏置策略：
- 每场随机交换 A/B 顺序。
- 支持 `TIE`。
- 强制严格 JSON 输出，便于自动解析。

### 3.3 使用你当前目录直接跑

默认已适配：
- `ocr_html_experiment_four_models/logs/records.json`
- `ocr_html_experiment_four_models/assets/images/*.jpg`
- `ocr_html_experiment_four_models/rendered_html/<model>/*.html`

执行：

```bash
# Windows PowerShell
$env:VLM_API_KEY="你的key"
$env:VLM_API_BASE="https://api.vectorengine.cn/v1"

python main.py arena-judge \
  --judge-model gpt-4o \
  --rounds 1 \
  --decisions-out outputs/arena_decisions.json

python main.py arena-elo \
  --decisions-json outputs/arena_decisions.json \
  --out outputs/arena_elo.json
```

输出：
- `outputs/arena_renders/`：每个样本每个模型的渲染图
- `outputs/arena_decisions.json`：逐场对战判决
- `outputs/arena_elo.json`：最终 Elo 排名

## 4. 目录结构

```text
benchmark/
  arena.py
  cli.py
  html_extractor.py
  render.py
  schema.py
  scorer.py
  utils.py
main.py
requirements.txt
ground_truth_schema_example.json
```

## 5. 说明

- 当前实现是工程化 baseline，重点是可运行和可扩展。
- 如果你后续给出固定的 GT 标注 schema，我可以再加一个更严格的匹配器（例如标签类型约束、匈牙利匹配、区域层级一致性等）。
