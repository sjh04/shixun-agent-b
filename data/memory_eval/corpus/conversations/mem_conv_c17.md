# 向量缓存设计

- memory_id: `mem_conv_c17`
- conversation_id: `eval_c17`
- created_or_updated_at: `2026-06-21T10:00:00+00:00`

## Final Answer

SQLite 检索缓存 memory_retrieval_cache.sqlite3 的作用是避免每次检索都重建 chunk 和重算 embedding。向量缓存键由记忆 id、chunk 序号、向量后端和内容 hash 四部分拼成，内容一变 hash 就变，旧向量自动失效不会被误用，因此不需要显式清缓存。qwen 和 hashing 两个后端的向量分开缓存互不覆盖。该数据库可随时删除，代价只是下次检索重建 chunk 并重算向量。
