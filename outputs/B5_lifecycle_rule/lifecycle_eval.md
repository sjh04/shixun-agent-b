# B5 生命周期评测（llm=off）

## 淘汰一致性（evict vs oracle）

- Kendall τ：1.000
- 淘汰集合与 oracle 一致：True
- 容量稳定：30 → 20（上限 20）
- pinned / global 保护违例：0

## 反思洞见

- 触发反思：2/3 簇
- 洞见命中率（洞见级查询进 top-k）：0.667
- 洞见全文见 lifecycle_eval.json 的 insight_documents_for_manual_scoring，供人工评分（1–5）。
