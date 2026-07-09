# selected_memory.json 字段约定

- memory_id: `mem_g_schema`
- created_or_updated_at: `2026-06-15T09:00:00+00:00`

## Insight

B1 的记忆注入只依赖三个基础字段：memory_id、memory_type、content，这三个字段永远不删不改。检索增强后新增的字段全部是可选的：score 是三因子总分，rank 是最终名次，factors 拆出相关度、时近性、重要性三个分量，retrieval_mode 记录检索模式，compressed 与 compression_method 标记压缩情况，flagged 表示被 Poison Gate 标记。消费方应当忽略不认识的字段，保证向前兼容。
