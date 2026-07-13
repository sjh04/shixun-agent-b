# Reflection b5294122ae36

- memory_id: `mem_global_reflect_b5294122ae36`
- created_or_updated_at: `2026-07-02T03:49:06+02:00`
- source_count: `3`
- compression_method: `cluster_join`

## Cluster

```json
{
  "method": "embedding_greedy",
  "backend": "hashing",
  "candidate_count": 4,
  "cluster_count": 2,
  "cluster_size": 3,
  "threshold": 0.05,
  "source_memory_ids": [
    "mem_conv_life_05",
    "mem_conv_life_06",
    "mem_conv_life_08"
  ]
}
```

## Insight

- OOM 修复方案
CUDA out of memory 用 bfloat16 加 max_memory 上限 20GiB 解决，float32 更占显存。
- 加载耗时实测
Qwen3.5-4B 模型加载约九十秒，进程内必须缓存复用，禁止重复加载。
- 显存检查习惯
共享服务器启动任务前先 nvidia-smi 看显存余量，避免挤崩别人的训练。
