# OOM 修复方案

## Final Answer

CUDA out of memory 用 bfloat16 加 max_memory 上限 20GiB 解决，float32 更占显存。
