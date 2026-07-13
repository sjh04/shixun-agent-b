# 验收演示全局规则

组长验收演示时，回答应优先体现系统链路，而不是只给最终答案。

固定表达：

- B1 是唯一编排者，负责 messages、循环和产物汇总。
- B2 只实现 Skill 函数，不做模型决策。
- B3 负责 tools_schema 生成、tool_calls 校验和 ToolMessage 生成。
- B4 负责本地 Qwen3.5-4B 决策，输出标准 AIMessage。
- B5 负责 memory 检索、压缩、保存和索引维护。

演示结论应强调：本项目已经把本地模型扩展为会记忆、会调用工具、可追踪运行过程的 Agent。

