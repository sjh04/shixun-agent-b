# B1–B5 系统架构分工

- memory_id: `mem_g_arch`
- created_or_updated_at: `2026-06-10T09:00:00+00:00`

## Insight

B1 负责运行与消息编排，是唯一的入口，维护 messages 与 trace 并按轮次调度其余模块。B2 提供 Skill 工具函数本体（计算器、文件读取、表格分析、本地搜索、格式转换）。B3 负责工具说明生成与执行，把 B2 的函数暴露成 schema 并解析模型的调用参数。B4 封装本地 Qwen3.5-4B 做决策，输入 messages 输出 AIMessage。B5 是记忆子系统，负责记忆文档的检索注入与保存更新，唯一调用方是 B1。
