# 最终交付说明（中文）

本仓库包含主观评测 v1 的正式实现。它不是实验脚手架，而是经过最终系统测试后固化的功能体。

核心改进包括：

- 将主观评测目标纠正为“人类可感知的页面视觉还原度”，不再人为追求主客观解耦；
- 新版 Prompt 明文位于 `benchmark/prompts/subjective_v1.md`，并通过 SHA256 固定；
- 七类 GT-driven 视觉轴用于关注和诊断，而不是硬权重加总；
- 每个 physical pair 强制 canonical + swapped 双方向评测；
- 双方向不一致转为 `reliability TIE`，与真正的 `perceptual TIE` 区分；
- 使用离线、顺序无关的 Batch Bradley-Terry 取代旧 sequential Elo；
- 使用 balanced round-robin、预渲染、请求体预检、reserve replacement、checkpoint/resume；
- 输出 bootstrap 区间和多种可靠性诊断。

最终实验表明：系统工程、可靠性机制和 Elo/BT 聚合均达到发布要求；无偏 30-pair 盲审没有证明新版 Prompt 的人类准确率高于旧 Prompt，因此最终对外表述应强调“评测规范与系统可靠性改进”，而不是声称“Prompt 准确率显著提升”。

正式实验报告见 `docs/FINAL_SYSTEM_VALIDATION_REPORT.md`；代码合并方式见 `docs/MIGRATION_FROM_ORIGINAL.md`。
