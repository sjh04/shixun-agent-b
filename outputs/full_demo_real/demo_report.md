# Full Agent Demo Report

- Conversation: `conv_001`
- Status: `success`
- Message flow: `system → user → assistant → tool → assistant`
- Tool rounds: `1`
- LLM calls: `2`
- Loaded memory documents: `4`
- Available tools: `5`

## Final Answer

1. Agent 系统通常由模型、工具、记忆和执行循环组成
2. 工具调用让模型能够读取本地文件、执行计算，并把结果用于后续回答
3. Memory 为 Agent 提供全局知识和历史对话上下文

## Output Files

- `final_answer.md`
- `llm_calls/llm_call_001_ai_message.json`
- `llm_calls/llm_call_001_raw_model_output.json`
- `llm_calls/llm_call_002_ai_message.json`
- `llm_calls/llm_call_002_raw_model_output.json`
- `llm_calls/llm_run_log.jsonl`
- `memory_log.jsonl`
- `messages.json`
- `runtime_log.jsonl`
- `saved_memory.json`
- `selected_memory.json`
- `tool_call_log.jsonl`
- `tool_messages.json`
- `tool_schema_report.json`
- `tools_schema.json`
- `trace.json`
