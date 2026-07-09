# 计算器浮点精度问题

- memory_id: `mem_conv_c01`
- conversation_id: `eval_c01`
- created_or_updated_at: `2026-06-18T10:00:00+00:00`

## Final Answer

用户问为什么计算器算 0.1 + 0.2 得到 0.30000000000000004。原因是二进制浮点数无法精确表示十进制小数。calculator 工具内部已改用 Decimal 做十进制运算，对外返回结果前按需要的小数位四舍五入，所以通过工具计算不会再出现这类尾数。直接在 Python 里用 float 仍会有该现象，属于语言特性不是 bug。
