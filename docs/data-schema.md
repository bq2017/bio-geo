# 数据格式

每条题目记录至少包含以下字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `item_id` | 是 | 稳定且唯一的题目编号 |
| `subject` | 是 | `biology` 或 `geography` |
| `grade` | 是 | 学段或年级 |
| `question_type` | 是 | 题型 |
| `stem` | 是 | 题干 |
| `options` | 否 | 选择题选项 |
| `answer` | 否 | 标准答案 |
| `explanation` | 否 | 答案解析 |
| `knowledge_tags` | 是 | 知识点标签列表 |
| `taxonomy_version` | 是 | 标签体系版本 |
| `annotation_status` | 是 | 标注状态 |

知识内容标签、能力标签和题目属性应使用不同字段管理，不能混入同一层知识树。

