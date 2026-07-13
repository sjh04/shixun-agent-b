# B5 优化回归评测

## Metadata Ablation

| 设置 | Hit@1 | Hit@3 | MRR | 平均延迟(ms) |
| --- | --- | --- | --- | --- |
| metadata off | 0.400 | 0.400 | 0.400 | 59.2 |
| metadata on | 0.800 | 0.800 | 0.800 | 33.6 |

## Cache / Update Invalidation

- 第二次相同 query cache：{'path': '/mnt/aisdata/sjh04/实训2/agent/outputs/B5_optimization/cache_update/memory/memory_retrieval_cache.sqlite3', 'hits': 1, 'misses': 0, 'disabled': 0}
- save-time prewarm：{'status': 'success', 'chunk_cache': {'hits': 0, 'misses': 1}, 'vector_backend': 'hashing', 'embedding_cache': {'embedding_cache_hits': 0, 'embedding_cache_misses': 2, 'embedding_cache_disabled': 0}}
- 更新前 SQLite 计数：{'documents': 5, 'chunks': 7, 'vectors': 7, 'query_vectors': 1}
- 更新后 SQLite 计数：{'documents': 5, 'chunks': 8, 'vectors': 8, 'query_vectors': 2}
- 更新后检索摘要：ImportError; ModuleNotFoundError; PyYAML; exec_command; python3; agent/configs/runtime.yaml; agent/configs/model.yaml; python3 -m pip install PyYAML
- 更新后文件 signals：['agent/configs/runtime.yaml', 'agent/configs/model.yaml']
