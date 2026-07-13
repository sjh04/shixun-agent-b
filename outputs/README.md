# 验收输出目录说明

保留用于验收和复现实验的结果：

- `B5_memory/`：B5 基础查找/保存 CLI 演示。
- `B5_ablation_hashing_opt/`：RQ1 检索消融，LLM off / hashing 兜底路径。
- `B5_ablation_qwen_opt/`：RQ1 检索 FULL，LLM on / Qwen 路径。
- `B5_optimization/`：本轮 save/load 优化回归，含 metadata ablation、cache 和更新失效测试。
- `B5_compression_rule/`、`B5_compression_qwen/`：RQ2 压缩层对照。
- `B5_compression_qwen_prompt_v1/`、`B5_compression_qwen_prompt_v2/`：RQ2 摘要 prompt 迭代留档，用于说明最终 prompt 选择依据。
- `B5_integration_rule/`、`B5_integration_qwen/`：RQ3/RQ4 整合与 Poison Gate 对照。
- `B5_lifecycle_rule/`、`B5_lifecycle_qwen/`：RQ5 生命周期评测。
- `B5_e2e/`：RQ6 端到端记忆效用评测。
- `full_demo_mock/`、`full_demo_real/`：全系统演示输出。
- `demo_flow_mock/`：新增现场演示主线输出，展示 B1+B3+B4+B5 完整闭环。
- `demo_b3_calculator/`、`demo_b3_table/`：新增现场演示工具变体输出，展示 B3 对固定 tool call 的执行。

旧版 ablation 输出和临时 verify 输出已清理；历史总结见 `../RESULTS_B5.md`。
