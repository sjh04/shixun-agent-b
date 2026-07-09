# 工具使用规范

- memory_id: `mem_g_tools`
- created_or_updated_at: `2026-06-12T09:00:00+00:00`

## Insight

calculator 接收算式字符串返回数值，内部用高精度计算，不要传自然语言。file_reader 只能读 data 根目录内的文本文件，路径越界会被拒绝。table_analyzer 针对 CSV 表格，支持求和、平均值、最大最小值等操作，需要指定列名。local_file_search 支持通配符模式匹配文件名，可选递归子目录。format_converter 在 markdown、html、json 等格式间转换文本。所有工具返回统一的status/errors 结构，调用失败不会中断主流程。
