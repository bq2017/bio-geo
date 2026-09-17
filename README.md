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

`geography-tagging-units.jsonl` 保留给后续调用二使用。调用一不读取展开后的
小题记录，而是直接读取每行一整道原题的聚合数据。

按完整原题随机抽取30道题用于候选召回小测试：

```bash
question-tagging-sample \
  --input data/processed/geography-merged-with-labels.jsonl \
  --catalog data/processed/geography-labels-call1-catalog.txt \
  --output data/annotation/geography-whole-question-sample-30.jsonl \
  --summary-output runs/tagging/geography-whole-question-sample-30-summary.json \
  --groups 30 \
  --seed 20260916
```

聚合数据中的每一行就是一道完整原题。普通题保留自身；大题在同一条记录中
保留公共题干和全部 `sub_questions`。没有可用题目内容、没有原标签或原标签
不属于当前414标签的题目不进入样本。正式抽取1000题时只需把文件名中的
`30` 和 `--groups 30` 改为 `1000`。

## 调用一：候选标签召回

调用一读取完整原题和414条简明标签目录。一道普通题调用一次完整流程；一道
大题把公共题干和全部小题作为一个整体调用一次完整流程，得到整道题共用的
候选集合。为避免超过模型上下文限制，脚本把目录均分为三批，分别召回后合并：

```bash
question-label-candidates \
  --input data/annotation/geography-whole-question-sample-30.jsonl \
  --catalog data/processed/geography-labels-call1-catalog.txt \
  --output runs/tagging/geography-call1-whole-question-sample-30-candidates.jsonl \
  --log-file runs/logs/geography-call1-whole-question-sample-30-candidates.log \
  --base-url http://172.22.0.35:9204/v1 \
  --model DeepSeek-V4-Flash \
  --concurrency 10
```

脚本不会把输入数据中已有的 `knw_labels` 发送给模型。候选标签必须是目录中的
完整路径，三批合并后最多20个且不要求凑满。合并结果超过20个时，脚本会让
模型根据原题从合并候选中收敛一次，不直接截断。运行中断后重复执行同一命令
即可跳过已完成的打标单元。

使用题目中已有的 `knw_labels` 检查候选召回：

```bash
question-label-candidates-evaluate \
  --input data/annotation/geography-whole-question-sample-30.jsonl \
  --candidates runs/tagging/geography-call1-whole-question-sample-30-candidates.jsonl \
  --catalog data/processed/geography-labels-call1-catalog.txt \
  --summary-output runs/tagging/geography-call1-whole-question-sample-30-evaluation-summary.json \
  --missing-output runs/tagging/geography-call1-whole-question-sample-30-missed-labels.jsonl
```

每条候选结果直接与同一道完整原题顶层的 `knw_labels` 比较，不再合并多个
打标单元，因此不使用 `--group-by-root`。汇总文件记录全量覆盖率和逐标签
召回率；明细文件记录漏召回标签以及无法与当前414个标签对应的原标签。

诊断调用一的漏召回和无关候选时，可以从完整原题中随机抽取100题，指定50道大题
和50道普通题，并保存每批召回及收敛前后的候选：

```bash
PYTHONPATH=src python -m bio_geo_tagging.sample_tagging_groups \
  --input data/processed/geography-merged-with-labels.jsonl \
  --catalog data/processed/geography-labels-call1-catalog.txt \
  --output data/annotation/geography-whole-question-sample-100.jsonl \
  --summary-output runs/tagging/geography-whole-question-sample-100-summary.json \
  --groups 100 --big-questions 50 --seed 20260917

PYTHONPATH=src python -m bio_geo_tagging.call1_candidate_retrieval \
  --input data/annotation/geography-whole-question-sample-100.jsonl \
  --catalog data/processed/geography-labels-call1-catalog.txt \
  --output runs/tagging/geography-call1-sample-100-candidates.jsonl \
  --trace-output runs/tagging/geography-call1-sample-100-trace.jsonl \
  --log-file runs/logs/geography-call1-sample-100.log \
  --base-url http://172.22.0.35:9204/v1 \
  --model DeepSeek-V4-Flash --concurrency 20
```

诊断文件逐题记录三批候选、收敛前候选以及是否执行收敛；原候选结果文件格式不变。
重复执行同一命令会跳过已完成且有诊断记录的题目。

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
  --input-jsonl data/processed/geography-merged-with-labels.jsonl \
  --definitions-jsonl data/taxonomy/geography-existing-definitions.jsonl \
  --output-jsonl runs/validation/geography-label-match.jsonl \
  --log-file runs/logs/geography-label-match.log \
  --base-url http://172.22.0.35:9102/v1 \
  --model DeepSeek-V4-Flash \
  --workers 1 \
  --limit 10
```

每条聚合记录作为一道完整大题，只评估大题自身 `knw_labels` 中的原标签。模型输入包含
公共题干和全部小题，不单独生成小题评分。输出 `score`、`reason`，并由代码按
`score >= 0.70` 计算 `match`。缺图无法判断时 `score` 和 `match` 均为
`null`。每次运行都会覆盖评分结果文件和运行日志，保证当前实验不混入历史记录。
本步骤不统计分数分布或 ABCD 分类。

运行日志会记录总任务数、每条任务的完成状态、当前进度、累计耗时和最终汇总，
可使用 `tail -f runs/logs/geography-label-match.log` 实时查看。

模型必须同时返回 `judgement`：正常评分为
`scored`，材料不足为 `unjudgeable`。代码会强制检查 `unjudgeable` 必须对应
`score=null`，避免将缺图题误记为0分。

### 阶段性异常知识点分析

先复制正在写入的评分结果作为阶段性快照，再按历史报告口径生成ABCD统计、异常
知识点JSONL和带完整题目示例的Markdown报告：

```bash
cp runs/validation/geography-label-match.jsonl \
  runs/validation/geography-label-match-40pct.jsonl

PYTHONPATH=src python -m bio_geo_tagging.label_match_analysis \
  --match-jsonl runs/validation/geography-label-match-40pct.jsonl \
  --questions-jsonl data/processed/geography-merged-with-labels.jsonl \
  --definitions-jsonl data/taxonomy/geography-existing-definitions.jsonl \
  --statistics-json runs/analysis/geography-label-statistics-40pct.json \
  --anomalies-jsonl runs/analysis/geography-label-anomalies-40pct.jsonl \
  --review-samples-jsonl runs/analysis/geography-label-review-samples-40pct.jsonl \
  --report-md runs/reports/geography-label-analysis-40pct.md \
  --scope-label "40%阶段性快照"
```

ABCD阈值为：A≥0.80、B为0.70～0.79、C为0.40～0.69、D<0.40。
默认将“至少5条且D级不少于3条、D级占比不低于30%”或“至少3条且全部为
D级”的知识点列为异常候选。另为所有已覆盖标签生成待诊断样本，每个标签最多
每个知识点包含最多10道A/B级高匹配题和10道C/D级低匹配题。每组优先选取5道
接近分级阈值的边界题，再从其余题目中固定抽取5道；样本可重复生成，但不代表该
知识点下的全部题目。它们用于同时检查释义过宽、过窄和
边界歧义；题量不足时保留全部可用题目。

将所有标签的待诊断样本逐条交给DeepSeek，判断问题能否通过修改释义解决：

```bash
PYTHONPATH=src python -m bio_geo_tagging.label_definition_diagnosis \
  --review-samples-jsonl runs/analysis/geography-label-review-samples-40pct.jsonl \
  --output-jsonl runs/analysis/geography-label-definition-diagnosis-40pct.jsonl \
  --log-file runs/logs/geography-label-definition-diagnosis-40pct.log \
  --base-url http://172.22.0.35:9204/v1 \
  --model DeepSeek-V4-Flash \
  --workers 1 \
  --limit 3
```

诊断会同时检查高匹配假阳性和低匹配假阴性，并将完全无关的历史误标单独列为
`unrelated_mislabel_ids`。每道高匹配样本必须归入有效匹配、释义过宽造成的假阳性
或模型误判之一；每道低匹配样本也必须归入一个诊断类别。对于确认存在释义
问题的知识点，结果中的 `definition_issue_evidence_questions` 会附带证据题目的完整
题干、选项、解析、小题、分数和第一阶段判断理由。先用少量标签验证结果，再去掉
`--limit` 执行阶段性全量。

诊断以完整大题为单位，并遵循“大题标签是公共题干与所有小题知识点并集”的规则。
只要任意一道小题直接考查当前知识点，就视为大题与该标签有效匹配；不得因该知识点
不是整道大题的主要主题而排除。即使知识点整体被判为 `insufficient_evidence`，所有
抽样题目也必须完成结构化分类，不允许遗漏。

全量完成后如存在 `error`，可在原命令末尾增加 `--retry-errors`。该模式保留所有
`completed` 结果，只重新处理错误或缺失标签，并用重试结果替换原错误行；输出文件
仍然每个标签一行，不会追加重复记录。

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

