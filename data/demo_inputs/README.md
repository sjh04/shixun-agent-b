# Agent 演示输入

这些输入用于现场展示完整 Agent 流程，均从 `agent/code` 目录运行。

- `demo_file_reader_memory.json`：主线演示。B1 调用 B5 加载记忆，B3 导出工具，B4 生成 tool call，B3 执行 `file_reader`，B4 生成最终回答，B5 保存本轮对话记忆。
- `demo_calculator.json`：真实模型模式下的 calculator 全链路输入。
- `demo_table_analyzer.json`：真实模型模式下的 table_analyzer 全链路输入。
- `ai_message_calculator_tool_call.json`：固定 calculator tool call，用于无 GPU 时直接演示 B3 工具执行。
- `ai_message_table_analyzer_tool_call.json`：固定 table_analyzer tool call，用于无 GPU 时直接演示 B3 工具执行。

`demo_file_reader_memory.json` 会写入 `memory/conversations/demo_flow_file_reader.md` 并更新 `memory/memory_index.json`。如果只想展示运行链路、不改正式记忆，可临时把 `save_memory` 改为 `none`。
