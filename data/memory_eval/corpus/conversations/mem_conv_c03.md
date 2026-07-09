# CUDA OOM 最新解决方案

- memory_id: `mem_conv_c03`
- conversation_id: `eval_c03`
- created_or_updated_at: `2026-06-28T10:00:00+00:00`

## Final Answer

推理时报 CUDA out of memory 的最新结论：第一，加载用 bfloat16 而不是 float32，显存直接省一半；第二，给 from_pretrained 传 max_memory 把单卡上限压到 20GiB，留出生成期激活的余量；第三，把 max_new_tokens 从 2048 降到 1024。三条一起用之后再没有出现过 OOM。注意：五月份那次结论（用 float32 加小 batch）已经过时作废，不要再参考。
