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

## 阶段一：混合候选召回

正式阶段一不调用DS。程序对每道完整题目同时执行字符级BM25和
`BAAI/bge-large-zh-v1.5`语义召回，并分别处理普通标签、区域标签和综合标签：

- 普通标签：BM25优先保留21个，再由BGE补充至30个；
- 区域标签：使用78个区域标签的名称、别称、包含地点和代表地点信息，最多保留5个；
- 综合标签：使用专用检索文本和加权RRF排序，最多保留5个；
- 每道题最终最多得到40个候选标签。

正式文件如下：

| 用途 | 文件 |
| --- | --- |
| 完整题目输入 | `data/annotation/geography-high-score-valid-questions-filtered-v2.jsonl` |
| 414个标签完整原释义 | `data/taxonomy/geography-existing-definitions.jsonl` |
| 78个区域标签元数据 | `data/taxonomy/geography-region-label-metadata.jsonl` |
| 普通及综合标签索引 | `data/processed/geography-label-retrieval-index-char2-comprehensive-bm25-v3/` |
| 区域标签索引 | `data/processed/geography-region-label-retrieval-index-phrase/` |
| 阶段一候选结果 | `runs/tagging/geography-stage1-final-candidates-1826.jsonl` |
| 阶段一评测汇总 | `runs/tagging/geography-stage1-final-summary-1826.json` |

正式运行命令：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60

PYTHONPATH=src .venv/bin/python -m bio_geo_tagging.hybrid_candidate_retrieval \
  --input data/annotation/geography-high-score-valid-questions-filtered-v2.jsonl \
  --index-dir data/processed/geography-label-retrieval-index-char2-comprehensive-bm25-v3 \
  --region-index-dir data/processed/geography-region-label-retrieval-index-phrase \
  --output runs/tagging/geography-stage1-final-candidates-1826.jsonl \
  --summary-output runs/tagging/geography-stage1-final-summary-1826.json \
  --nonregion-candidate-limit 30 \
  --region-candidate-limit 5 \
  --comprehensive-candidate-limit 5 \
  --region-bm25-min-score 0 \
  --region-bge-min-score 0.4 \
  --embedding-model BAAI/bge-large-zh-v1.5 \
  --device cpu \
  --batch-size 32
```

区域标签元数据属于标签体系的检索补充信息；由它生成的区域检索文本和索引属于
可重建的中间文件。第二阶段不读取区域元数据或召回分数，只从阶段一结果中提取
`question_id`、`parent_id`和`combined_candidates[].label_path`，再关联414个标签的
完整原释义。

## 第一、二阶段统一运行

正式运行可使用统一入口依次生成打标单元、执行候选召回、运行DS精排、生成自动诊断，
并合并旧标签与DS新增标签：

```bash
PYTHONPATH=src .venv/bin/python -m bio_geo_tagging.run_tagging_pipeline \
  --input data/annotation/geography-high-score-valid-questions-filtered-v2.jsonl \
  --labels data/taxonomy/geography-existing-definitions.jsonl \
  --index-dir data/processed/geography-label-retrieval-index-char2-comprehensive-bm25-v3 \
  --region-index-dir data/processed/geography-region-label-retrieval-index-phrase \
  --run-dir runs/tagging/geography-full-pipeline-v1 \
  --audited-exclusions configs/geography_adjudication_audited_exclusions.json \
  --endpoint http://172.22.0.35:9204/v1/chat/completions \
  --model DeepSeek-V4-Flash \
  --disable-thinking \
  --nonregion-candidate-limit 30 \
  --region-candidate-limit 5 \
  --comprehensive-candidate-limit 5 \
  --region-bm25-min-score 0 \
  --region-bge-min-score 0.4 \
  --embedding-model BAAI/bge-large-zh-v1.5 \
  --device cpu \
  --batch-size 32 \
  --workers 30 \
  --timeout 600 \
  --retries 3 \
  --retry-delay 1 \
  --request-interval 0 \
  --max-tokens 512
```

统一运行目录包含：

```text
geography-full-pipeline-v1/
├── tagging-units.jsonl
├── candidates.jsonl
├── candidate-summary.json
├── adjudication/
│   ├── evidence.jsonl
│   ├── predictions.jsonl
│   ├── question_predictions.jsonl
│   ├── report.json
│   └── run.log
├── evaluation.json
├── evaluation_details.jsonl
├── final_labels.jsonl
├── pipeline.log
├── pipeline_manifest.json
└── pipeline_report.json
```

相同命令和运行目录可断点续跑：已完成的打标单元和候选召回会跳过，DS精排复用
`evidence.jsonl`中的成功记录。输入、索引路径、模型或关键参数变化时必须使用新的
`--run-dir`。任何阶段失败都会停止后续阶段，不能用不完整结果生成最终标签。
统一入口不会把召回文件中的`knw_labels`或评测字段发送给DS。原来的阶段一、阶段二
独立命令继续保留，用于单独调试。

## 第二阶段：候选标签精排

第二阶段从第一阶段候选中选择当前题目或当前小题直接考查的知识点。精排程序兼容：

- 混合召回输出中的 `combined_candidates`；
- DS 调用一输出中的 `candidate_labels`；
- 已规范化的 `candidates`。

完整标签释义文件至少要提供 `label_path` 和 `definition`。也可以直接使用现有地理字段
`positive_definition`/`assessment_scope`，或
`knowledge_scope`/`common_exam_content`/`distinction_from_similar_labels`。
正式地理释义文件也可使用 `knw_label` 和嵌套的 `existing_interpretation`；其中
`definition`、`keywords`、`exam_methods`、`distinction`会映射到精排字段，数字
`label_id`作为taxonomy ID保留。
建议正式运行前为全部标签补齐相邻标签边界。

整题召回结果可以直接用于小题精排：程序优先按小题查找候选，找不到时按
`root_question_id` 复用整题候选。DS请求不会提交图片或图片URL；如果上游已有可靠的
文字图片描述，重新生成打标单元后会通过 `context_image_description` 传给小题。
题目明确依赖图片但没有文字描述时，Prompt会要求模型标记
`context_insufficient=true`，不得根据答案或解析猜测图片内容。

召回文件只作为候选索引，不作为题目正文输入。程序仅提取记录的
`question_id`/`parent_id`和`combined_candidates[].label_path`，不会读取或发送
`knw_labels`、各种`missing_labels`、BM25/BGE分数与排名、`selection_source`或
区域证据字段。题目正文来自独立生成的打标单元；候选路径会先做与召回顺序无关的
确定性打乱，再关联标签文件中的完整释义后发送给DS。

普通题执行一次全部候选精排。大题的每个小题同时判断普通标签和区域标签，公共题干
只作上下文；区域仅作为材料发生地或定位信息时不选。公共题干和全部小题另执行一次
综合Label专项判断。最终 `question_predictions.jsonl` 将各小题的普通/区域标签与
整题综合标签合并为整道题结果。

示例：

```bash
PYTHONPATH=src python -m bio_geo_tagging.run_candidate_adjudication \
  --units data/annotation/geography-high-score-valid-tagging-units-filtered-v2.jsonl \
  --candidates runs/tagging/geography-stage1-final-candidates-1826.jsonl \
  --labels data/taxonomy/geography-existing-definitions.jsonl \
  --audited-exclusions configs/geography_adjudication_audited_exclusions.json \
  --run-dir runs/tagging/geography-adjudication-smoke-50-v1.5 \
  --endpoint http://172.22.0.35:9204/v1/chat/completions \
  --model DeepSeek-V4-Flash \
  --disable-thinking \
  --limit 50 \
  --workers 10 \
  --timeout 600 \
  --retries 3 \
  --retry-delay 1 \
  --request-interval 0 \
  --max-tokens 512
```

输出目录包含 `evidence.jsonl`、`predictions.jsonl`、`report.json`、
`question_predictions.jsonl`、`run_manifest.json` 和 `run.log`。`run.log`会追加记录
每次启动、逐单元`OK/ERROR`进度、错误原因和最终汇总，可用`tail -f`实时查看；
逐单元进度不再重复打印到终端，终端只显示启动信息和最终摘要。
`predictions.jsonl`是逐小题/逐综合专项结果，`question_predictions.jsonl`是整道题最终并集。
整题结果同时记录 `expected_component_count`、`completed_component_count`、
`components_complete` 和 `missing_component_units`；任一预期单元未成功时，整题不可训练。
输入文件、模型、Prompt或限制发生变化时，必须使用新的运行目录。
`--limit`按整道题计数；大题的综合专项和全部小题不会被拆开截断。

打标完成后可离线生成自动诊断。程序不会调用DS，也不会把现有标签加入Prompt：

```bash
PYTHONPATH=src python -m bio_geo_tagging.adjudication_evaluation \
  --run-dir runs/tagging/geography-adjudication-smoke-100-v1.5 \
  --legacy-candidates runs/tagging/geography-stage1-final-candidates-1826.jsonl \
  --labels data/taxonomy/geography-existing-definitions.jsonl
```

输出 `evaluation.json`、`evaluation_details.jsonl` 和 `final_labels.jsonl`。诊断按
`final_labels = legacy_labels ∪ ds_added_labels` 计算最终标签，重点报告DS新增规模、
普通/区域/综合标签分布、层级冗余、evidence结构校验和自动风险标记。
`final_labels.jsonl`可直接供下游使用，并分别保留旧标签、DS新增标签和最终并集。
原有Precision、Recall、F1和完全匹配率保留在`legacy_label_agreement`中，
仅表示DS与现有标签的一致程度，不表示新增标签的语义准确率。
缺少任一预期组件的整题只计入 `incomplete_question_predictions` 和
`attempted_questions_without_complete_prediction`，不进入自动诊断统计。
未进行人工新增标签裁决时，`semantic_addition_accuracy_available=false`，不得把自动诊断
表述为准确率。需要比较两次独立运行的稳定性时，可增加
`--stability-run-dir <另一运行目录>`。

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

### 仅根据标签名称复评已有题目—标签对

以旧评分结果作为唯一任务清单，取回相同完整大题，仅向模型提供知识点完整路径
名称，不提供现有释义。输出顺序与旧评分任务顺序一致，便于后续按照
`(question_id, label_id)` 对比两种评分：

```bash
PYTHONPATH=src python -m bio_geo_tagging.question_label_match score-by-name \
  --pairs-jsonl runs/validation/geography-label-match-40pct.jsonl \
  --questions-jsonl data/processed/geography-merged-with-labels.jsonl \
  --output-jsonl runs/validation/geography-label-name-match-40pct.jsonl \
  --log-file runs/logs/geography-label-name-match-40pct.log \
  --base-url http://172.22.0.35:9204/v1 \
  --model DeepSeek-V4-Flash \
  --workers 25 \
  --timeout 180 \
  --limit 10
```

小批量验证后去掉 `--limit 10` 运行全部旧任务。命令不会读取释义文件，也不会根据
题目当前的 `knw_labels` 增加任务；默认会覆盖新结果文件和日志，不影响旧的释义评分
结果。若任务中断，续跑时加上 `--resume`：程序保留输出文件中已完成的
`(question_id, label_id)`，跳过这些任务，并重新处理错误或未完成的任务；日志仍会覆盖
写入本次运行。

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

