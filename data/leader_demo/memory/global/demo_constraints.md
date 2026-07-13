# 现场演示约束

验收现场优先使用低风险、稳定工具：

- `file_reader`：读取本地 Markdown 或 txt 文件。
- `calculator`：执行确定性数学计算。
- `local_file_search`：在指定目录搜索关键词。
- `table_analyzer`：分析 CSV / TSV 表格。
- `format_converter`：将文本转换为 Markdown 或 JSON 文件。

不建议在默认演示中开放高风险代码执行工具。`code_executor` 与 `composite` 可作为 B2 进阶原型说明，但不放入默认 `basic_tools`。

