# 本地文件通配符搜索

- memory_id: `mem_conv_c06`
- conversation_id: `eval_c06`
- created_or_updated_at: `2026-06-14T10:00:00+00:00`

## Final Answer

要找出目录下所有 markdown 文件，用 local_file_search 传模式 *.md 并打开递归开关，工具会遍历全部子目录返回相对路径列表。模式匹配只作用于文件名不含路径，想按路径过滤要在返回结果里自己筛。搜索范围被限制在 data 根目录内，越界返回错误。
