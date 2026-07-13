# code 目录说明

`agent/code` 根目录只保留系统运行核心入口：

- `b1_agent_runtime.py`：B1 Agent 总控运行与消息管理。
- `b1_batch_runner.py`：B1 批量任务运行器。
- `b2_run_skill.py`：B2 Skill 独立命令行入口。
- `b3_tool_layer.py`：B3 tools schema 生成与 tool call 执行。
- `b4_local_agent_llm.py`：B4 本地 LLM 决策模块。
- `b5_memory.py`：B5 记忆加载、检索、保存与管理。
- `run_full_demo.py`：完整 Agent 一键演示。
- `common/`：公共 IO、路径、日志和 schema 工具。

B5 评测/测试跑批脚本已统一放在 `evals/`。从本目录执行时，命令形式为：

```bash
python evals/run_b5_ablation.py --llm off --outdir ../outputs/B5_ablation_hashing
```
