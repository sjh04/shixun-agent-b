# 记忆截断问题

- memory_id: `mem_conv_c10`
- conversation_id: `eval_c10`
- created_or_updated_at: `2026-06-13T10:00:00+00:00`

## Final Answer

记忆内容太长被截断的处理：注入预算由 memory.yaml 的 max_memory_chars 控制，默认 2000 字符，多条记忆按名次先后瓜分预算。单条超预算时不再从中间硬切，而是调用压缩层生成摘要，保关键信息弃细节；模型不可用时退化为抽取式摘要。如果发现注入的记忆缺了关键内容，优先调大 max_memory_chars 而不是关压缩。
