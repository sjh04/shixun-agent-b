# JSONDecodeError 修复

- memory_id: `mem_conv_c16`
- conversation_id: `eval_c16`
- created_or_updated_at: `2026-06-19T10:00:00+00:00`

## Final Answer

json.loads 报 JSONDecodeError: Expecting value: line 1 column 1 的根因是待解析文本开头混入了模型输出的说明文字或日志前缀，不是 JSON 本身坏了。修复：解析前先定位第一个左花括号、截取到匹配的右花括号再 loads；同时在 prompt 里禁止模型输出 JSON 以外的字符。空字符串也会报同样的错，要先判空给出更明确的错误信息。
