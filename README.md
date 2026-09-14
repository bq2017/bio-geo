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
  --log-file runs/logs/geography-question-info-merge.log
```

该命令保持既有清洗和聚合规则，仅将输入、输出及日志路径改为运行时参数。

## 知识点释义分析

第一步仅根据完整知识点路径生成模型释义：

```bash
label-definition-generate \
  --input-xlsx data/taxonomy/geography-knowledge-graph.xlsx \
  --output-jsonl runs/validation/label-definition-generation.jsonl \
  --base-url http://172.22.0.35:9092/v1
```

确认第一步全部完成后，第二步读取生成结果并与 Excel 中的现有释义比较：

```bash
label-definition-compare \
  --input-xlsx data/taxonomy/geography-knowledge-graph.xlsx \
  --generated-jsonl runs/validation/label-definition-generation.jsonl \
  --output-jsonl runs/validation/label-definition-comparison.jsonl \
  --base-url http://172.22.0.35:9092/v1
```

两个阶段的结果分别逐条追加到 JSONL，重复执行时会跳过该阶段已经成功完成的知识点。如果第一步存在缺失结果，第二步会直接停止并报告缺失数量。
第二阶段只判断是否存在实质理解差异、具体差异和是否无法确定边界；需要教师复核的标记由代码根据“存在差异且无法确定”自动生成。

## 数据安全边界

- 不向仓库提交真实题目、答案、解析和人工标注数据。
- 不向仓库提交预测结果、评测结果和运行日志。
- 提交前使用 `git status` 检查待提交文件。

