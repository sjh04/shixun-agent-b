# Reflection e3d7d6804059

- memory_id: `mem_global_reflect_e3d7d6804059`
- created_or_updated_at: `2026-07-02T03:51:26+02:00`
- source_count: `4`
- compression_method: `qwen_reflection`

## Cluster

```json
{
  "method": "embedding_greedy",
  "backend": "qwen",
  "candidate_count": 4,
  "cluster_count": 1,
  "cluster_size": 4,
  "threshold": 0.35,
  "source_memory_ids": [
    "mem_conv_life_01",
    "mem_conv_life_02",
    "mem_conv_life_03",
    "mem_conv_life_04"
  ]
}
```

## Insight

系统采用混合检索架构，以 BM25 处理文本特征（k1=1.5, b=0.75，中文按字符 bigram 切词）与向量检索并行，通过 RRF 融合（rrf_k=60）合并结果并限制候选上限为 30 条。长记忆数据按 700 字符切块、120 字符重叠，检索后回溯原记忆上下文。最终仅对相关性前 10 名候选执行三因子重排，确保仅返回 top 5 条记忆，实现高效精准的长上下文检索。
