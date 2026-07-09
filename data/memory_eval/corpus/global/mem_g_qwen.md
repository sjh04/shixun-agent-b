# Qwen3.5-4B 加载配置

- memory_id: `mem_g_qwen`
- created_or_updated_at: `2026-06-08T09:00:00+00:00`

## Insight

模型权重放在 models/Qwen3.5-4B，加载参数固定为：torch_dtype 用 bfloat16（这张卡上float16 偶发溢出警告），device_map 设 auto 让权重自动分配到空闲显卡，local_files_only设 true 保证离线可用，trust_remote_code 设 true。tokenizer 与模型同路径。加载一次约九十秒，进程内要缓存复用，不要重复加载。
