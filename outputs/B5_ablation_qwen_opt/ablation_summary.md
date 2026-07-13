# B5 检索层消融矩阵（llm=on）

| 配置 | Hit@1 | Hit@3 | Hit@5 | MRR | nDCG@5 | 平均延迟(ms) | HyDE | rerank | 向量后端 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FULL | 0.950 | 1.000 | 1.000 | 0.975 | 0.982 | 6392.9 | qwen | qwen | qwen |

## 分探针类型 Hit@1（每个组件在其针对的查询类型上证明价值）

| 配置 | chunk_buried | keyword_exact | multi_relevant | paraphrase | three_factor |
| --- | --- | --- | --- | --- | --- |
| FULL | 1.00 (n=2) | 1.00 (n=4) | 1.00 (n=2) | 0.91 (n=11) | 1.00 (n=1) |
