# 高中生物地理知识点打标

本项目用于管理高中生物、地理题目的知识点打标代码。Git 仓库只保存代码、配置模板、测试和规范文档；原始数据、标注数据及运行结果仅保存在本地。

## 目录结构

```text
.
├── configs/                 # 可提交的配置模板
├── docs/                    # 架构、数据格式和标注规范
├── src/bio_geo_tagging/     # Python 源码
├── tests/                   # 自动化测试
├── data/                    # 本地数据，不提交 Git
└── runs/                    # 本地运行结果，不提交 Git
```

本地数据按以下阶段存放：

```text
data/raw → data/interim → data/annotation → data/processed
```

每次运行应在 `runs/` 下创建独立目录，并记录代码提交版本、数据版本、标签体系版本和运行配置。

## 本地配置

复制 `configs/local.example.yaml` 为 `configs/local.yaml`，再按本机环境修改。`local.yaml` 已被 Git 忽略。

## 开发环境

项目要求 Python 3.11 或更高版本：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

## 数据预处理

题目清洗和大小题聚合命令：

```powershell
question-info-merge `
  --input data/raw/geography.jsonl `
  --output data/processed/geography-merged.jsonl `
  --log-file runs/logs/geography-question-info-merge.log `
  --taxonomy data/taxonomy/high-geography-knw-leaf-nodes-0909.json
```

该命令保持既有清洗和聚合规则，并将原始 `knw_ids` 映射为
`knw_labels`。输出不保留 `knw_ids`。

将聚合结果展开为独立打标单元：

```powershell
question-tagging-units `
  --input data/processed/geography-merged.jsonl `
  --output data/annotation/geography-tagging-units.jsonl `
  --log-file runs/logs/geography-question-tagging-units.log
```

根题目和每个小题分别形成一条记录。小题记录通过 `context_stem`
携带大题公共题干，并保留该小题自己的 `knw_labels`。

## 调用一：候选标签召回

调用一读取独立打标单元和414条简明标签目录。为避免超过模型上下文限制，
脚本把目录均分为三批，对同一道题分别召回后合并候选：

```bash
question-label-candidates \
  --input data/annotation/geography-tagging-units.jsonl \
  --catalog data/processed/geography-labels-call1-catalog.txt \
  --output runs/tagging/geography-call1-candidates.jsonl \
  --log-file runs/logs/geography-call1-candidates.log \
  --base-url http://172.22.0.35:9204/v1 \
  --model DeepSeek-V4-Flash \
  --concurrency 10
```

脚本不会把输入数据中已有的 `knw_labels` 发送给模型。候选标签必须是目录中的
完整路径，三批合并后最多20个且不要求凑满。合并结果超过20个时，脚本会让
模型根据原题从合并候选中收敛一次，不直接截断。运行中断后重复执行同一命令
即可跳过已完成的打标单元。

## 原标签与原释义匹配评分

先从释义对比结果中提取**现有释义**（不使用 DS 生成释义）：

```bash
question-label-match export-definitions \
  --comparison-jsonl runs/validation/label-definition-comparison.jsonl \
  --output-jsonl data/taxonomy/geography-existing-definitions.jsonl
```

先测试少量题目—标签对，再去掉 `--limit` 跑全量：

```bash
question-label-match score \
  --input-jsonl data/annotation/geography-tagging-units.jsonl \
  --definitions-jsonl data/taxonomy/geography-existing-definitions.jsonl \
  --output-jsonl runs/validation/geography-label-match.jsonl \
  --base-url http://172.22.0.35:9102/v1 \
  --model DeepSeek-V4-Flash \
  --workers 1 \
  --limit 10
```

每个打标单元的每个原标签单独评估。输出 `score`、`reason`，并由代码按
`score >= 0.70` 计算 `match`。缺图无法判断时 `score` 和 `match` 均为
`null`。已成功处理的题目—标签对在重跑时跳过；错误记录在重跑时再次尝试。
只输出评分结果，不在本步骤统计分数分布或 ABCD 分类。

## 知识点释义分析

第一步仅根据完整知识点路径生成模型释义：

```bash
label-definition-generate \
  --input-xlsx data/taxonomy/geography-knowledge-graph.xlsx \
  --output-jsonl runs/validation/label-definition-generation.jsonl \
  --base-url http://172.22.0.35:9093/v1 \
  --secondary-base-url http://172.22.0.35:9104/v1 \
  --workers 4
```

确认第一步全部完成后，第二步读取生成结果并与 Excel 中的现有释义比较：

```bash
label-definition-compare \
  --input-xlsx data/taxonomy/geography-knowledge-graph.xlsx \
  --generated-jsonl runs/validation/label-definition-generation.jsonl \
  --output-jsonl runs/validation/label-definition-comparison.jsonl \
  --base-url http://172.22.0.35:9092/v1
```

两个阶段的结果分别逐条追加到 JSONL，重复执行时会跳过该阶段已经成功完成的知识点。第一阶段可将工作线程平均分配到两个接口。如果第一步存在缺失结果，第二步会直接停止并报告缺失数量。
第二阶段只输出是否存在实质理解差异、具体差异、是否需要教师复核和判断理由。

## 数据安全边界

- 不向仓库提交真实题目、答案、解析和人工标注数据。
- 不向仓库提交预测结果、评测结果和运行日志。
- 提交前使用 `git status` 检查待提交文件。

