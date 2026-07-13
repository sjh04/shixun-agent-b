# 检索演示材料

本文件用于演示 `local_file_search` 与 memory 检索的区别。

## 文件检索关键词

- 工具编排
- ReAct 循环
- ToolMessage
- memory 注入

## 核心事实

本地 Agent 的运行主线是：

1. B5 先根据用户问题加载相关记忆。
2. B1 将系统提示词、记忆和用户输入组装成 messages。
3. B4 根据 messages 和 tools_schema 决定是否调用工具。
4. B3 校验 tool_calls 并调用 B2 Skill。
5. B1 将 ToolMessage 写回 messages，再次调用 B4。
6. 当 AIMessage 不再包含 tool_calls 时，B1 输出 final_answer。

