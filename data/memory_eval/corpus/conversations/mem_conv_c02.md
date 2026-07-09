# CSV 列平均值统计

- memory_id: `mem_conv_c02`
- conversation_id: `eval_c02`
- created_or_updated_at: `2026-06-20T10:00:00+00:00`

## Final Answer

统计 CSV 某一列的平均值要用 table_analyzer 工具：传入表格路径、操作类型 mean、以及目标列名。列名必须与表头完全一致，含空格时要原样传入。数值列里混有空值时工具会自动跳过空值再求平均，行为与 Excel 一致。求和用 sum、最大最小用 max/min，同一次调用只支持一个操作。
