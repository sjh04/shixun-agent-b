# B5 记忆模块实验结果总览

对应 proposal《B5 记忆文档存储与查找模块》6.2 节 RQ1–RQ6。所有实验可复现：
语料与标注由 `code/evals/build_b5_eval_corpus.py` 及 `data/memory_eval/*.json` 固化，
跑批脚本见 `code/evals/run_b5_*.py`（用法见 README 5.5–5.9 节），每次运行的配置开关
与依赖版本快照自动写入各输出目录的 `memory_log.jsonl`。
环境：Python 3.10 / torch 2.7.1+cu118 / transformers 5.12.1 / 本地 Qwen3.5-4B（bfloat16）。

## 结论速览（proposal 6.6 成功判据对照）

| RQ | 判据 | 实测 | 结果 |
| --- | --- | --- | --- |
| RQ1 检索 | FULL 对 B0 的 Hit@3 显著提升；MRR 对最佳单路 +0.05–0.10；组件在对应探针上有增益 | Hit@3 0.20→1.00；MRR 0.877→0.975（+0.098）；各探针见下 | ✅ |
| RQ2 压缩 | 压缩比 ≤0.4；下游答对率 ≥1.5× 硬截断 | 0.333；0.600 vs 0.267（2.25×） | ✅ |
| RQ3 整合 | 三分类准确率 ≥85%，冲突检出 ≥90% | Qwen judge 100% / 100%（规则 38.9% / 16.7%） | ✅ |
| RQ4 投毒 | 拦截率 ≥80% 且误杀率 ≤10% | TPR 90% / FPR 10%（规则 0% / 40%） | ✅ |
| RQ5 生命周期 | evict 与 oracle 的 Kendall τ ≥0.8；容量稳定 | τ=1.000，30→20 零违例；反思命中率 qwen 100% | ✅ |
| RQ6 端到端 | 有记忆比无记忆任务成功率 +≥20 个百分点 | 0% → 100%（+100pp），检索命中率 100% | ✅ |

待人工环节：反思洞见 1–5 评分（材料在 `outputs/B5_lifecycle_qwen/lifecycle_eval.json`）、
摘要关键点保留的人工抽检校准（token 自动口径系统性低估同义换写）。

## 本轮优化回归（save/load 缓存与检索 metadata）

本轮新增 `code/evals/run_b5_optimization_eval.py`，专门验证保存侧
`retrieval_summary` / `retrieval_terms`、SQLite `query_vectors` 缓存、save-time
prewarm 以及 memory 更新后的缓存失效。结果落盘在
`outputs/B5_optimization/optimization_eval.{json,md}`。

| 实验 | 设置 | Hit@1 | Hit@3 | MRR | 结论 |
| --- | --- | --- | --- | --- | --- |
| Metadata ablation | metadata off | 0.400 | 0.400 | 0.400 | 只看正文时，文件名/错误码/命令/工具名类问题召回不足 |
| Metadata ablation | metadata on | 0.800 | 0.800 | 0.800 | 保存时抽取 trace/messages signals 后，检索显著提升 |

缓存与更新一致性：

- 第二次相同 query：`query_embedding_cache` 命中 1、miss 0；
- save-time prewarm：保存后已预热 chunk 与 hashing 向量，首次 load 可直接命中；
- memory 更新后：SQLite 计数从 `documents=5, chunks=7, vectors=7, query_vectors=1`
  变为 `documents=5, chunks=8, vectors=8, query_vectors=2`，新文件 signals
  `agent/configs/runtime.yaml` 进入索引，说明 source hash 与 query cache 均按新内容更新。

同时重跑 RQ1：

- `outputs/B5_ablation_hashing_opt/`：llm=off 完整矩阵，FULL = Hit@1 0.700 / Hit@3 0.950 / MRR 0.835；
- `outputs/B5_ablation_qwen_opt/`：llm=on FULL，确认走 Qwen backend、HyDE、rerank，
  FULL = Hit@1 0.950 / Hit@3 1.000 / MRR 0.975 / nDCG@5 0.982。

## RQ1 检索层消融（20 标注查询 × 24 记忆语料）

Qwen 路径（最新回归结果：`outputs/B5_ablation_qwen_opt/`；完整矩阵留档见旧目录）：

| 配置 | Hit@1 | Hit@3 | Hit@5 | MRR | nDCG@5 |
| --- | --- | --- | --- | --- | --- |
| B0 基础版（按 id 取） | 0.050 | 0.200 | 0.300 | 0.139 | 0.163 |
| KW（BM25） | 0.800 | 0.950 | 1.000 | 0.877 | 0.907 |
| VEC（Qwen 向量） | 0.850 | 0.900 | 0.900 | 0.875 | 0.874 |
| RRF 混合 | 0.850 | 0.950 | 0.950 | 0.892 | 0.903 |
| + chunk | 0.850 | 0.950 | 0.950 | 0.892 | 0.903 |
| + HyDE | 0.950 | 1.000 | 1.000 | 0.975 | 0.978 |
| + 三因子 | 0.850 | 1.000 | 1.000 | 0.917 | 0.934 |
| FULL（+ rerank） | **0.950** | **1.000** | **1.000** | **0.975** | **0.982** |

分探针 Hit@1（Qwen 路径）：改述类 KW 0.73 → VEC 0.82 → HyDE/FULL 0.91；
新旧冲突探针 KW–RRF 全部被过时记忆骗到（0.00），三因子与 FULL 修正为 1.00。

关键发现：

1. **三因子乘法公式缺陷**：proposal 原式 `rel × rec × imp` 实测使 Hit@1 从 0.80 崩至
   0.05——RRF 相关度（rank 倒数）取值范围过窄，被 rec×imp 淹没。修复为
   `w·norm(原始分) + w·rec + w·imp`（0.8/0.1/0.1）+ 相关度前 10 名门控，
   proposal 4.3 节已同步修订。
2. **三因子的取舍是结构性的**：在纯主题查询上有意让"新且重要"的近似平局者胜出
   （Hit@1 0.95→0.85），换来冲突纠偏；**下游 rerank 把这部分损伤完全修复**
   （FULL 回到 0.95 且保留冲突纠偏），构成"三因子+rerank"的组合论证。
3. **hashing 兜底 vs Qwen 向量**：hashing 路（最新回归结果：`outputs/B5_ablation_hashing_opt/`）FULL 仅
   0.700/0.950/0.835，改述类 0.45——语义召回是 Qwen embedding 的独有贡献；
   HyDE/rerank 在 llm off 时自动回退、零增益零损伤。
4. 延迟：BM25 毫秒级；Qwen 向量冷启动 ~20s/查询（缓存后 ~350ms）；HyDE +~4s；
   FULL ~7.7s/查询——精度与延迟的权衡按场景经 `memory.yaml` 选择。

## RQ2 压缩层（5 长样本 × 关键点清单 × 下游 QA，预算=min(700, 原文/3)）

| 方法 | 保留率(宽松) | 保留率(严格) | 压缩比 | 下游答对率 |
| --- | --- | --- | --- | --- |
| 硬截断（基础版） | 0.333 | 0.267 | 0.333 | 0.267 |
| 抽取式（无 GPU 兜底） | 0.233 | 0.233 | 0.302 | 0.267 |
| **Qwen 生成式（prompt v3）** | **0.700** | **0.667** | 0.333 | **0.600** |

摘要 prompt 经三轮迭代（各版结果留档 `outputs/B5_compression_qwen_prompt_v1/_v2`）：
v1 泛化表述 0.533/0.467 → v2 英文笔记式指令**负优化** 0.333/0.267（模型把中文记忆
译成英文笔记，关键点与中文问答全部失配）→ v3 = v1 + 逐字保留标识符 + **语言一致性
约束** 0.700/0.600。教训：4B 模型的摘要指令必须显式锁定输出语言。

## RQ3 整合三分类（18 标注样例：重复/补充/冲突各 6）

| 模式 | 准确率 | 冲突检出率 |
| --- | --- | --- |
| 纯规则（相似度+否定词） | 0.389 | 0.167 |
| **Qwen judge** | **1.000** | **1.000** |

规则路径把大量换写重复判成补充、无否定词的事实反转判不出冲突；Qwen judge 全对。
双路架构下 judge 失败自动回退规则，离线可用性不受影响。

## RQ4 Poison Gate（10 投毒 + 10 正常，对照 6 条全局可信记忆）

| 模式 | 拦截率 TPR | 误杀率 FPR |
| --- | --- | --- |
| 纯规则 | 0.000 | 0.400 |
| **Qwen NLI** | **0.900** | **0.100** |

规则路径完全失效且反向误杀（否定词启发在正常表述上误触发）；Qwen NLI 达标
（漏网 1 例、误杀 1 例，明细见 `outputs/B5_integration_qwen/integration_eval.json`）。

## RQ5 生命周期（合成 30 条超容量 + 3 簇反思语料）

- **淘汰**：`_evict_if_needed` 淘汰顺序与独立复算 oracle（重要性×时近性升序）
  Kendall τ = **1.000**，集合完全一致；容量 30→20 稳定；pinned/global 零违例。
- **反思**：3 簇同主题对话按簇触发 reflect——Qwen 路径 3/3 簇生成洞见、
  洞见级查询检索命中率 **1.000**；hashing 兜底路径 2/3、0.667。
  附带发现：hashing 短文本余弦系统性低于 Qwen embedding，反思聚类阈值须按后端
  取值（0.35 vs 0.05），否则兜底路径永不触发反思。

## RQ6 端到端记忆效用（10 个"只有靠记忆才能答对"的任务）

| 条件 | 任务成功率 |
| --- | --- |
| 无记忆 | 0.000（10/10 回答"不知道"或编造） |
| **B5 完整管线注入** | **1.000** |

检索命中率 100%（标注记忆全部进入 top-k）。本实验用"检索注入+单轮回答"近似
全系统链路；平均步数/重复提问率在 `run_full_demo` 多轮联调中另行演示。
