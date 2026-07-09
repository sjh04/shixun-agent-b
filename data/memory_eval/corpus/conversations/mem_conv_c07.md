# markdown 转 html 的坑

- memory_id: `mem_conv_c07`
- conversation_id: `eval_c07`
- created_or_updated_at: `2026-06-16T10:00:00+00:00`

## Final Answer

markdown 转 html 用 format_converter，目标格式参数必须显式传 html。两个坑：一是表格属于扩展语法，简单管道表格能转，嵌套表格会原样输出；二是行内 HTML 默认会被转义，需要保留标签时加 raw 选项。转出来的 html是文档片段不含 head，要完整页面得自己包一层。
