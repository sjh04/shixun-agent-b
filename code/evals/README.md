# B5 评测与测试脚本

本目录存放 B5 相关语料构建、评测和优化回归脚本。建议仍从 `agent/code` 目录执行，保持配置和输出路径写法一致。

| 脚本 | 用途 |
| --- | --- |
| `build_b5_eval_corpus.py` | 构建 B5 检索评测语料和标注查询。 |
| `evaluate_b5_memory.py` | 对查询集合计算 Hit@k、MRR、nDCG@5 和延迟。 |
| `run_b5_ablation.py` | RQ1 检索消融矩阵。 |
| `run_b5_compression_eval.py` | RQ2 压缩层评测。 |
| `run_b5_integration_eval.py` | RQ3/RQ4 更新整合与 Poison Gate 评测。 |
| `run_b5_lifecycle_eval.py` | RQ5 生命周期与 reflection 评测。 |
| `run_b5_e2e_eval.py` | RQ6 端到端记忆效用评测。 |
| `run_b5_optimization_eval.py` | save/load 优化回归测试。 |

示例：

```bash
cd agent/code
python evals/build_b5_eval_corpus.py
python evals/run_b5_ablation.py --llm off --outdir ../outputs/B5_ablation_hashing
python evals/run_b5_optimization_eval.py --outdir ../outputs/B5_optimization
```
