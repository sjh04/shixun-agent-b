# CUDA out of memory 排查记录（旧）

- memory_id: `mem_conv_c04`
- conversation_id: `eval_c04`
- created_or_updated_at: `2026-05-03T10:00:00+00:00`

## Final Answer

显卡内存不够、推理报 CUDA out of memory 的排查记录。当时的现象是加载即崩溃，报错出现在 from_pretrained 阶段。临时结论：把 torch_dtype 改成 float32 并把 batch 压到 1，勉强能跑但极慢；实在不行重启机器碰运气。（注：这是早期笔记，后来发现 float32 反而更占显存，本结论已被六月末的新方案取代，仅留档。）
