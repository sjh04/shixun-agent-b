# 团队项目 README

> 实训 B 方向 · 本地文件驱动的 Agent 智能体框架
>
> 本 README 按团队项目模板组织，覆盖：项目解决什么问题、系统如何运行、依赖哪些模型与数据、如何复现完整演示与各模块演示、以及团队如何把 B1–B5 五个模块组合成一个完整系统。

---

## 1. 项目概述

### 1.1 项目名称

`本地 Agent 框架（实训 B 方向）`

### 1.2 项目目标

面向"本地、离线、文件驱动"的 Agent 任务场景，基于服务器本地的 **Qwen3.5-4B** 大模型，构建一个可运行、可观测、可复现的 Agent 智能体框架。系统要解决的核心问题是：让一个纯语言模型具备 **记忆、工具调用、多轮循环控制和长期状态管理** 的能力，把"模型回答一句话"扩展为"模型自主决策 → 调用工具 → 读取记忆 → 汇总结果 → 保存记忆"的完整闭环。

最终实现的核心能力：

- 完整消息序列编排：`system → user → assistant(tool_calls) → tool → assistant(final)`；
- 5 个基础 Skill（计算、读文件、本地检索、表格分析、格式转换）；
- 工具说明（tools_schema）自动生成、参数校验、工具执行、重试 / 缓存 / 统计；
- 本地 LLM 决策（真实 `prompt_json` 模式 + 无 GPU `mock` 调试模式）；
- 主动记忆管理（关键词 / 向量 / 混合检索、压缩、整合、投毒拦截、生命周期淘汰与反思）；
- 全链路结构化产物与运行日志，支持一键复现。

### 1.3 当前完成情况

| 类型 | 完成情况 |
|---|---|
| 基础要求 | **B1–B5 基础要求全部完成**：独立模块、统一接口、结构化输出、全链路联调，均满足 PPT 要求。 |
| 进阶要求 | **B1 / B3 / B5 完成度高**：B1 多轮对话、断点续跑、批量任务、历史摘要压缩、Prompt 切换、plan / confirm 模式；B3 有限重试、结果缓存、调用统计；B5 检索增强、压缩、整合、投毒拦截、生命周期。**B2 / B4 部分完成**：B2 扩展 Skill（代码执行、复合 Skill）已实现但未接入默认工具集；B4 完成本地决策与两种运行模式，模型切换 / 原生工具绑定对比实验不足。 |
| 支持的主要任务类型 | 数学计算、本地 txt/md 文档阅读与摘要、本地文件检索、CSV/TSV 表格分析、文本格式转换（markdown/json）、以及无工具的直接问答。 |
| 当前限制 | B4 缺少不同模型 / 原生 tools 传参的系统化对比实验；B2 的 `code_executor`、`composite` 尚未接入默认 `basic_tools`；B3 尚未实现"从 Python 函数自动生成完整 schema"；Qwen 向量检索冷启动延迟较高（首查询 ~20s，缓存后 ~350ms）。 |

---

## 2. 整体流程与模块结构

### 2.1 模块边界

系统由 5 个模块（B1–B5）加 1 个一键演示入口组成。**B1 是唯一的运行时编排者**，B2–B5 均可独立命令行运行，也可被 B1 通过公开函数编排。模块之间只通过三种标准 JSON 契约传递数据：`SkillResult`、`AIMessage`、`ToolMessage`（见附录 A）。

| 模块 / 阶段 | 入口文件 / 入口函数 | 主要职责 | 输入 | 输出 |
|---|---|---|---|---|
| **B1 Agent Runtime** | `code/b1_agent_runtime.py` | Agent 总控：消息管理、循环控制（`max_turns`）、编排 B3/B4/B5、产物汇总 | 用户问题 + `configs/*.yaml` | `messages.json` / `trace.json` / `final_answer.md` |
| **B2 Skill 层** | `code/b2_run_skill.py`（实现在 `skills/`） | 独立执行 5 个基础 Skill，封装错误码 | Skill 的 JSON 输入 | `SkillResult`（JSON） |
| **B3 Tool 层** | `code/b3_tool_layer.py` | 生成 `tools_schema`，校验并执行 `tool_calls`（内部调用 B2 Skill） | `configs/tools.yaml` + `tool_calls` | `tools_schema.json` / `tool_messages.json` |
| **B4 LLM 决策** | `code/b4_local_agent_llm.py` | 用本地 Qwen 生成标准 `AIMessage`（只决策，不执行工具） | `messages` + `tools_schema` | `AIMessage`（工具调用或最终回答） |
| **B5 Memory** | `code/b5_memory.py` | 记忆检索 / 压缩 / 整合 / 投毒拦截 / 保存 / 索引维护 | `configs/memory.yaml` + `query` | `selected_memory.json` / `saved_memory.json` |
| **完整演示** | `code/run_full_demo.py` | 以 B1 为入口跑通全链路并生成汇总报告 | 同 B1 | 全部 integrated 产物 + `demo_report.md` |

### 2.2 系统架构图 / 流程图

整体数据流（B1 编排，B3 内部调用 B2）：

```text
                            ┌──────────────────────────────┐
        用户问题 query ────▶ │        B1 Agent Runtime       │
                            │  （消息管理 + 循环控制 max_turns）│
                            └──────┬─────────────┬──────────┘
                                   │             │
                    ①取记忆         │             │  ②取工具说明 tools_schema
                                   ▼             ▼
                        ┌───────────────┐   ┌───────────────┐
                        │   B5 Memory   │   │  B3 Tool 层   │
                        │ 检索/压缩/整合 │   │ schema/校验/执行│
                        └───────┬───────┘   └───────┬───────┘
                                │                   │ 调用
              memory context    │                   ▼
                                │           ┌───────────────┐
                                │           │   B2 Skill    │
                                │           │ 5 个基础工具   │
                                │           └───────┬───────┘
                                │                   │ SkillResult
                                ▼                   ▼
        messages ─────▶ ┌──────────────────────────────────┐
        (system/user/   │           B4 LLM 决策             │
         ai/tool)       │   本地 Qwen3.5-4B → AIMessage     │
                        └──────────────────────────────────┘
                                   │
             AIMessage 有 tool_calls？──是──▶ 回到 B3 执行工具（循环）
                                   │否
                                   ▼
                    final_answer.md  +  ③B5 保存本轮对话记忆
```


### 2.3 一次完整任务的流程

以默认演示任务 `data/runtime_input.json`（"读取 `docs/agent_intro.txt` 并用三条中文要点总结"）为例：

1. **原始输入**：用户问题 + `configs/`（model / tools / memory）+ `prompts/local_tool_agent.txt` 系统提示模板。
2. **记忆注入**：B1 调用 B5，按检索管线从 `memory/` 取出相关记忆，拼进初始 `messages`。
3. **首次决策**：B1 调用 B3 生成 `tools_schema`，连同 `messages` 交给 B4；本地 Qwen 输出一个带 `tool_calls` 的 `AIMessage`（决定调用 `file_reader`）。
4. **工具执行**：B1 把 `tool_calls` 交给 B3，B3 校验参数后调用 B2 的 `file_reader` Skill，读取 `data/docs/agent_intro.txt`，返回 `ToolMessage`。
5. **二次决策**：B1 把 `ToolMessage` 追加进 `messages` 再交给 B4，Qwen 基于工具结果生成最终回答（`tool_calls=[]`、`content` 非空）。
6. **收尾与保存**：B1 写出 `final_answer.md`，并调用 B5 把本轮 `messages/trace/final_answer` 保存为对话记忆，更新 `memory/memory_index.json`。
7. **日志与产物**：全过程写出 `messages.json`、`trace.json`、各类 `*_log.jsonl`、`llm_calls/` 原始输出等，`run_full_demo.py` 额外汇总 `demo_report.md`。

---

## 3. 模型、数据集与外部资源

### 3.1 模型说明

| 项目 | 内容 |
|---|---|
| 使用模型 | **Qwen3.5-4B**（本地权重，通过 `transformers` 直接加载，bfloat16） |
| 模型来源 | 服务器本地已有权重 / 官方模型页面下载 |
| 项目内相对路径 | `models/Qwen3.5-4B/`，由 `configs/model.yaml` 的 `model_name_or_path` / `tokenizer_name_or_path` 指定（默认相对路径 `../models/Qwen3.5-4B`，相对 `configs/` 解析）；`models/` 已 `.gitignore`，**不随源码包分发** |
| 是否需要 GPU | 真实 `prompt_json` 模式**需要 GPU**；`mock` 调试模式**不需要** |
| 是否需要联网运行 | **不需要**（完全本地离线运行） |

放置模型（下载到指定目录，或对已有权重建软链）：

```bash
# 方式一：把权重目录放到 models/Qwen3.5-4B/
#   models/Qwen3.5-4B/{config.json, *.safetensors, tokenizer.*, ...}

# 方式二：对服务器上已有权重建软链
ln -s /path/to/your/Qwen3.5-4B  models/Qwen3.5-4B

# 方式三：改用绝对路径 —— 编辑 configs/model.yaml 的 model_name_or_path / tokenizer_name_or_path
```

### 3.2 数据集 / 示例数据说明

本项目数据均为**项目自带**或**由脚本可复现构造**，无需外部下载。源码包按要求不含 `data/`、`models/`、`memory/`、`outputs/`，运行前请从团队仓库获取完整 `data/` 与 `memory/`，或用下方脚本重新生成评测语料。

| 数据或文件 | 用途 | 来源 | 项目内相对路径 |
|---|---|---|---|
| Skill 输入样例 | B2 各 Skill 的正常 / 异常输入 | 项目自带 | `data/tool_inputs/` |
| 消息样例 | B3/B4 的 tool_calls 与 messages 样例 | 项目自带 | `data/messages/` |
| B1 fixtures | B1 个人演示的预设 memory/AI/Tool 响应 | 项目自带 | `data/b1_fixtures/` |
| runtime 任务输入 | 全系统 integrated 任务输入 | 项目自带 | `data/runtime_input*.json` |
| 记忆保存样例 | B5 保存对话 / 全局记忆的输入 | 项目自带 | `data/memory_inputs/` |
| B5 评测语料与标注 | RQ1–RQ6 的标注 query 与语料 | 脚本可复现构造 | `data/memory_eval/` |
| 演示文档 | file_reader / 检索实际读取的文档 | 项目自带 | `data/docs/` |
| 记忆库 | 全局 / 对话记忆文档与索引 | 项目自带 + 运行时更新 | `memory/`（`global/`、`conversations/`、`memory_index.json`） |

复现 B5 评测语料（可选，评测前执行）：

```bash
cd agent/code
python evals/build_b5_eval_corpus.py   # 生成 data/memory_eval/corpus/ 与 corpus_queries.json
```

---

## 4. 环境安装

### 4.1 运行环境

| 项目 | 要求 |
|---|---|
| Python 版本 | Python 3.10 |
| 操作系统 / 服务器环境 | Linux 服务器（已在 CUDA 11.8 环境验证） |
| GPU 要求 | 真实模式需 GPU（加载 Qwen3.5-4B，bfloat16）；无 GPU 可用 `mock` 模式跑通全链路调试 |
| 主要依赖 | `torch==2.7.1+cu118`、`transformers==5.12.1`、`accelerate`、`PyYAML`、`sentencepiece`、`safetensors`、`numpy`（完整见 `requirements.txt`） |

### 4.2 安装步骤

推荐每位同学新建独立 conda 环境：

```bash
# 1. 克隆 / 获取项目后进入 agent 目录
cd agent

# 2. 创建并激活环境
conda create -n agent python=3.10 -y
conda activate agent
export PYTHONNOUSERSITE=1   # 禁止加载用户级 site-packages，保证只用当前环境的包

# 3. 安装依赖（torch 使用 CUDA 11.8 wheel）
pip install -r requirements.txt

# 4. 放置模型（见 3.1 节），随后所有演示命令都从 code 目录执行
cd code
```

常见环境问题：

- **模型路径不存在**：`configs/model.yaml` 默认指向 `../models/Qwen3.5-4B`。请确认权重已放好或已建软链，或改为绝对路径；无 GPU / 无模型时先用 `--llm_mode mock` / `--mode mock` 验证链路。
- **依赖版本不兼容**：`torch` 需带 `+cu118` 后缀，务必保留 `--extra-index-url`；`export PYTHONNOUSERSITE=1` 可避免串用用户级旧包。
- **GPU 显存不足**：确认使用 bfloat16、`device_map: auto`；必要时降低 `max_new_tokens`，或改用 `mock` 模式做流程演示。

---

## 5. 输入文件与配置文件说明

### 5.1 主要配置文件

| 配置文件 | 作用 | 需要修改的字段 |
|---|---|---|
| `configs/model.yaml` | 本地模型配置（Qwen3.5-4B、transformers、bf16、`prompt_json`） | `model_name_or_path` / `tokenizer_name_or_path`（模型路径）、`max_new_tokens`、`default_mode` |
| `configs/tools.yaml` | 定义 toolset、每个工具的模块/函数、参数、必填项、`data_root`、重试/缓存开关 | `toolset` 选择、`settings.max_retries`、`cache_enabled`、工具级 `retryable` / `cacheable` |
| `configs/memory.yaml` | 记忆根目录、检索管线（none/keyword/vector/hybrid）、压缩、整合、生命周期、`max_memory_chars` | 检索模式与各增强开关、`max_memory_chars`、`lifecycle.*` |
| `configs/memory_small_limit.yaml` | 复用当前 `memory/`，仅调小 `max_memory_chars` 用于截断演示 | `max_memory_chars` |
| `configs/memory_*acceptance*.yaml` / `memory_leader_demo.yaml` / `memory_eval_corpus.yaml` | 验收 / 评测专用记忆配置变体 | 按演示场景选择 |

### 5.2 主要输入文件

| 输入文件 | 用途 | 适用场景 |
|---|---|---|
| `data/runtime_input.json` | file_reader 主线任务（读取文档 + 三条中文要点） | 完整系统 / 一键 Demo |
| `data/runtime_input_0.json` | 无工具倾向任务，验证模型直接回答 | 完整系统 |
| `data/runtime_input_2.json` ~ `_5.json` | 分别对应 calculator / local_file_search / table_analyzer / format_converter 任务 | 完整系统（分工具） |
| `data/b1_fixtures/b1_fixture_input.json` | B1 个人演示入口（预设 memory/AI/Tool 响应，不调 B2–B5） | 模块演示（B1 隔离） |
| `data/tool_inputs/tool_input_*.json` | 5 个 Skill 的正常输入；`*_error.json` 为异常样例 | 模块演示（B2）/ 异常样例 |
| `data/messages/ai_message_with_tool_calls.json` 等 | B3 工具执行样例（正常 / 未知工具 / 缺参 / 重试 / 缓存） | 模块演示（B3）/ 异常样例 |
| `data/messages/messages_no_tool.json` / `messages_with_tool.json` / `messages_with_error_tool.json` | B4 两阶段决策与失败工具处理输入 | 模块演示（B4） |
| `data/memory_inputs/memory_save_*.json` | B5 保存对话 / 全局记忆输入 | 模块演示（B5） |
| `data/memory_eval/*.json` | RQ1–RQ6 标注语料与查询 | 评测 |

---

## 6. 完整流程 Demo 运行

> 所有命令均从 `agent/code` 目录执行。既提供**一键完整 Demo**（第 6.2 节 A），也保留 **B1–B5 各模块的独立演示**（第 6.2 节 B），两者并重。

### 6.1 Demo 样例说明

| Demo | 输入文件 / 输入内容 | 演示目的 |
|---|---|---|
| **一键完整 Demo** | `data/runtime_input.json` | 以 B1 为入口，真实调用 B3/B4/B5 跑通 `system→user→ai(tool)→tool→ai(final)` 全链路并生成汇总报告 |
| B1 个人演示（fixture） | `data/b1_fixtures/b1_fixture_input.json` | 用预设响应隔离验证 B1 的消息管理与循环控制，不依赖其他模块 |
| B2 Skill 演示 | `data/tool_inputs/tool_input_*.json` | 独立验证 5 个基础 Skill 的输入/输出/错误封装 |
| B3 Tool 演示 | `data/messages/*.json` | 验证 schema 生成、参数校验、工具执行、重试 / 缓存 / 统计 |
| B4 LLM 演示 | `data/messages/messages_*.json` | 验证本地 Qwen 生成工具调用型 / 最终回答型 AIMessage |
| B5 Memory 演示 | 命令行 `--select_memory_ids` / `--save_input_path` | 验证记忆检索、截断、保存与索引更新 |

### 6.2 运行命令

**A. 一键完整 Demo（推荐，真实模型）**

```bash
cd agent/code
python run_full_demo.py \
  --input ../data/runtime_input.json \
  --tools_config ../configs/tools.yaml \
  --memory_config ../configs/memory.yaml \
  --model_config ../configs/model.yaml \
  --llm_mode prompt_json \
  --outdir ../outputs/full_demo
```

> 无 GPU / 无模型时，把 `--llm_mode prompt_json` 改为 `--llm_mode mock` 即可跑通全链路（不加载模型、不占显存）。
> 该命令会按 `runtime_input.json` 的 `save_memory=conversation` 更新 `memory/conversations/conv_001.md` 与 `memory/memory_index.json`，重复演示前请确认可覆盖。

**B. B1–B5 各模块独立演示**

```bash
# ── B1 个人演示（fixture，不调用 B2–B5）
python b1_agent_runtime.py --input ../data/b1_fixtures/b1_fixture_input.json --outdir ../outputs/B1_fixture

# ── B1 全系统 integrated（以 B1 为入口真实调用 B3/B4/B5；此处 LLM 用 mock）
python b1_agent_runtime.py --input ../data/runtime_input.json \
  --tools_config ../configs/tools.yaml --memory_config ../configs/memory.yaml \
  --model_config ../configs/model.yaml --llm_mode mock --outdir ../outputs/B1_runtime

# ── B2 五个基础 Skill（逐个）
python b2_run_skill.py --skill calculator        --input ../data/tool_inputs/tool_input_calculator.json     --outdir ../outputs/B2_skills
python b2_run_skill.py --skill file_reader       --input ../data/tool_inputs/tool_input_file_reader.json     --outdir ../outputs/B2_skills
python b2_run_skill.py --skill local_file_search --input ../data/tool_inputs/tool_input_file_search.json     --outdir ../outputs/B2_skills
python b2_run_skill.py --skill table_analyzer    --input ../data/tool_inputs/tool_input_table_analyzer.json  --outdir ../outputs/B2_skills
python b2_run_skill.py --skill format_converter  --input ../data/tool_inputs/tool_input_format_converter.json --outdir ../outputs/B2_skills

# ── B3 生成 schema / 执行 tool_calls / 进阶（重试·缓存·统计）
python b3_tool_layer.py --tools_config ../configs/tools.yaml --toolset basic_tools --export_schema --outdir ../outputs/B3_tools
python b3_tool_layer.py --tools_config ../configs/tools.yaml --toolset basic_tools --tool_calls ../data/messages/ai_message_with_tool_calls.json --execute --outdir ../outputs/B3_tools
python b3_tool_layer.py --tools_config ../configs/tools.yaml --toolset demo_tools  --tool_calls ../data/messages/b3_tool_call_retry_recoverable.json --execute --outdir ../outputs/B3_tools/retry
python b3_tool_layer.py --tools_config ../configs/tools.yaml --toolset basic_tools --tool_calls ../data/messages/b3_tool_call_cache_repeat.json --execute --outdir ../outputs/B3_tools/cache

# ── B4 本地 LLM 决策（真实 prompt_json：第一阶段生成 tool_call、第二阶段生成 final_answer）
python b4_local_agent_llm.py --model_config ../configs/model.yaml --messages ../data/messages/messages_no_tool.json  --tools_schema ../data/messages/tools_schema_basic.json --mode prompt_json --outdir ../outputs/B4_llm/no_tool_real
python b4_local_agent_llm.py --model_config ../configs/model.yaml --messages ../data/messages/messages_with_tool.json --tools_schema ../data/messages/tools_schema_basic.json --mode prompt_json --outdir ../outputs/B4_llm/with_tool_real

# ── B5 记忆查找 / 保存
python b5_memory.py --config ../configs/memory.yaml --select_memory_ids mem_conversation_conv_000 --use_global_memory true --query "Agent 系统如何调用工具？" --outdir ../outputs/B5_memory
python b5_memory.py --config ../configs/memory.yaml --save_type conversation --save_input_path ../data/memory_inputs/memory_save_input.json --outdir ../outputs/B5_memory
```

> B5 六个研究问题（RQ1–RQ6）的完整评测命令见 **附录 C**。

### 6.3 关键参数说明

| 参数 | 说明 |
|---|---|
| `--input` | 任务输入 JSON（B1 / full_demo），决定用户问题、执行模式与 `save_memory` 策略 |
| `--tools_config` / `--memory_config` / `--model_config` | 分别指向 `tools.yaml` / `memory.yaml` / `model.yaml` |
| `--llm_mode`（B1/full_demo）、`--mode`（B4） | `prompt_json`=加载本地模型真实运行；`mock`=不加载模型的调试模式 |
| `--outdir` | 输出目录，全部结构化产物与日志写入此处 |
| `--skill` / `--input`（B2） | 选择 Skill 及其 JSON 输入 |
| `--toolset` / `--export_schema` / `--execute`（B3） | 选择工具集、导出 schema、执行 tool_calls |
| `--select_memory_ids` / `--use_global_memory` / `--query` / `--save_type` / `--save_input_path`（B5） | 控制记忆查找与保存 |
| `--resume`（B1） | 从 checkpoint 断点续跑（进阶） |

### 6.4 运行成功的判断方式

- 终端显示运行完成且**无 traceback**，CLI 退出码为 `0`（退出码含义见附录 B）。
- 输出目录生成主要结果文件：`messages.json`、`trace.json`、`final_answer.md`，一键 Demo 还会生成 `demo_report.md`。
- `messages.json` 的角色顺序符合规范链路：`system → user → assistant(tool_calls) → tool → assistant(final)`。
- `final_answer.md` 含预期的最终回答（如主线任务的三条中文要点）；`trace.json` 中 `status` 正常、工具轮次与 LLM 次数符合预期。

---

## 7. 输出文件与结果说明

### 7.1 主要输出文件

一键完整 Demo（`outputs/full_demo/`）运行后的关键产物：

| 输出文件 | 生成模块 / 阶段 | 格式 | 说明 |
|---|---|---|---|
| `messages.json` | B1 | JSON 数组 | 完整 Agent 消息序列（顶层固定为数组） |
| `trace.json` | B1 | JSON 对象 | 运行状态、工具轮次、LLM 次数、每轮消息、memory 保存状态与错误 |
| `final_answer.md` | B1 | Markdown | 最终给用户的回答 |
| `selected_memory.json` | B5（B1 调用） | JSON 对象 | 注入前选择的记忆及截断结果 |
| `saved_memory.json` | B5（B1 调用） | JSON 对象 | 本轮对话记忆的保存结果与目标路径 |
| `tools_schema.json` / `tool_messages.json` | B3（B1 调用） | JSON 数组 | 本轮工具 schema 与产生的 ToolMessage |
| `tool_call_log.jsonl` / `runtime_log.jsonl` / `memory_log.jsonl` | B3 / B1 / B5 | JSONL | 工具执行、运行时、记忆操作的累计日志 |
| `llm_calls/llm_call_00N_*.json` | B4（B1 调用） | JSON 对象 | 每次 LLM 调用的原始输出与规范化 AIMessage |
| `demo_report.md` | `run_full_demo.py` | Markdown | 汇总对话、数据流、工具/LLM 次数、最终回答与文件清单 |
| `memory/conversations/conv_001.md`、`memory/memory_index.json` | B5 | Markdown / JSON | 保存本轮对话记忆并更新索引（写入项目正式记忆目录） |

各模块独立演示的产物目录（`outputs/B1_fixture/`、`B2_skills/`、`B3_tools/`、`B4_llm/`、`B5_memory/`）及每个文件的详细含义见**附录 D**。

### 7.2 运行截图或结果图例

B5 记忆模块六个研究问题（RQ1–RQ6）的实测结果（详见 `RESULTS_B5.md`）：

| RQ | 能力 | 关键指标 | 结果 |
|---|---|---|---|
| RQ1 检索 | 检索管线消融 | Hit@3 | **0.20 → 1.00**；MRR 0.877 → 0.975 |
| RQ2 压缩 | 同预算下游答对率 | 答对率 | **0.60 vs 硬截断 0.267（2.25×）** |
| RQ3 整合 | 三分类准确率 | 准确率 | **Qwen judge 100%**（规则 38.9%） |
| RQ4 投毒拦截 | Poison Gate | TPR / FPR | **90% / 10%**（规则 0% / 40%） |
| RQ5 生命周期 | 淘汰顺序 vs oracle | Kendall τ | **1.000**，30→20 零违例 |
| RQ6 端到端 | 依赖记忆任务成功率 | 成功率 | **0/10 → 10/10（+100pp）** |

---

## 8. 协作实现说明

从工程协作角度，团队通过"**统一契约 + 隔离演示 + 配置驱动**"把五个模块拼成一个完整系统：

- **约定统一的模块 I/O 契约**：全系统只用三种 JSON 对象跨模块传递数据 —— `SkillResult`（B2↔B3）、`AIMessage`（B4→B1）、`ToolMessage`（B3→B1），字段与形态在附录 A 固定，任何模块只要产出/消费这三种结构即可对接。
- **用配置文件和样例数据降低联调成本**：`tools.yaml` / `memory.yaml` / `model.yaml` 把模块行为参数化；`data/` 下为每个模块准备了正常与异常输入样例，各模块可独立命令行验证后再联调。
- **用 mock / fixture 解耦对硬件与彼此的依赖**：B4 的 `mock` 模式让无 GPU / 无模型的同学也能跑通链路；B1 的 `fixture` 模式用预设响应隔离验证编排逻辑，不依赖 B2–B5 的真实实现，使各模块可并行开发。
- **处理数据格式不一致**：B3 依据 Skill 函数签名自动注入 `data_root` / `output_dir`，屏蔽文件类 Skill 的路径差异；B5 在保存侧抽取 trace/messages 信号写入检索 metadata，弥合"正文检索"与"结构化查询"的差异。
- **可复现与可观测**：每个模块运行都追加结构化 `*_log.jsonl`；B5 每次 load/save 还快照配置开关与依赖版本，评测语料由脚本固化生成，保证结果可复现。
- **需多模块配合才能完成的能力**：完整的 `LLM→Tool→LLM` 闭环（B1+B3+B4）、带记忆注入与保存的全链路（+B5）、以及一键 Demo 汇总报告，均为多模块协同产物。

---

## 9. 已知问题与改进方向

| 问题 | 当前原因 | 可能改进 |
|---|---|---|
| B4 缺少模型 / 工具绑定对比实验 | 目前只接入本地 Qwen3.5-4B，且工具绑定统一走 prompt 注入 | 增加多本地模型可切换配置，补充"原生 tools 传参 vs prompt 注入"、工具调用成功率与 token 用量的批量对比 |
| B2 扩展 Skill 未接入默认工具集 | `code_executor`、`composite` 属进阶扩展，未加入 `basic_tools` | 在 `tools.yaml` 中正式接入并补充沙箱/超时的对外文档与样例 |
| B3 schema 未做到完全自动生成 | 目前基于 `tools.yaml` 生成 schema | 实现从 Python 函数签名/docstring 自动生成完整 `tools_schema`，并做"schema 描述质量对工具调用准确率影响"的对比实验 |
| Qwen 向量检索冷启动延迟高 | 首次查询需加载 embedding（~20s），FULL 管线 ~7.7s/查询 | 已加 SQLite/chunk/向量缓存（缓存后 ~350ms）；可进一步做 save-time 预热与按场景在 `memory.yaml` 选择精度/延迟档位 |
| 摘要/反思含人工评分环节 | token 自动口径系统性低估同义换写；反思洞见需人工打分 | 引入语义级保留率评估；补充人工抽检校准流程 |

---

## 附录

### 附录 A：公共数据格式（模块间契约）

**SkillResult**（B2 一次 Skill 执行；失败时 `status=error`、`output=null`、`error` 含异常类型与信息）：

```json
{ "skill_name": "calculator", "status": "success",
  "input": {"expression": "23 * 17 + 9"}, "output": {"result": 400},
  "error": null, "latency_ms": 0.5 }
```

**AIMessage**（B4 输出）—— 工具调用型 `content=""` 且 `tool_calls` 非空；最终回答型 `content` 非空且 `tool_calls=[]`：

```json
{ "role": "assistant", "content": "",
  "tool_calls": [{"id": "call_001", "name": "file_reader",
                  "args": {"path": "docs/agent_intro.txt", "max_chars": 2000}}] }
```

**ToolMessage**（B3 输出）—— `content` 是序列化后的 SkillResult 字符串，`tool_call_id` 关联对应 AIMessage：

```json
{ "role": "tool", "tool_call_id": "call_001", "name": "file_reader",
  "content": "{\"skill_name\":\"file_reader\",...}", "status": "success" }
```

### 附录 B：CLI 退出码

| 退出码 | 含义 |
|---|---|
| 0 | 成功，或业务错误已被捕获并写入结构化产物（异常样例也返回 0） |
| 1 | 配置 / 输入文件 / 解析 / 模块加载 / 模型依赖 / 输出目录等致命错误 |
| 2 | argparse 参数使用错误 |

### 附录 C：B5 记忆模块评测命令（RQ1–RQ6）

```bash
cd agent/code
python evals/build_b5_eval_corpus.py                                            # 构造标注语料

# RQ1 检索消融（--llm off 为无 GPU 兜底；--llm on 走 Qwen embedding + HyDE + rerank）
python evals/run_b5_ablation.py       --llm off --outdir ../outputs/B5_ablation_hashing
python evals/run_b5_ablation.py       --llm on  --outdir ../outputs/B5_ablation_qwen
# RQ2 压缩
python evals/run_b5_compression_eval.py --llm off --outdir ../outputs/B5_compression_rule
python evals/run_b5_compression_eval.py --llm on  --outdir ../outputs/B5_compression_qwen
# RQ3 整合三分类 / RQ4 Poison Gate（--judge off 纯规则；--judge on 走 Qwen judge/NLI）
python evals/run_b5_integration_eval.py --judge off --outdir ../outputs/B5_integration_rule
python evals/run_b5_integration_eval.py --judge on  --outdir ../outputs/B5_integration_qwen
# RQ5 生命周期（淘汰 + 反思）
python evals/run_b5_lifecycle_eval.py --llm off --outdir ../outputs/B5_lifecycle_rule
python evals/run_b5_lifecycle_eval.py --llm on  --outdir ../outputs/B5_lifecycle_qwen
# RQ6 端到端记忆效用（需 GPU）
python evals/run_b5_e2e_eval.py --outdir ../outputs/B5_e2e
# 检索指标（Hit@1/3/5、MRR、nDCG@5、平均延迟）
python evals/evaluate_b5_memory.py --config ../configs/memory.yaml --queries ../data/memory_eval/b5_eval_queries.json --outdir ../outputs/B5_eval
```

### 附录 D：各模块独立演示输出

| 目录 | 关键文件 | 说明 |
|---|---|---|
| `outputs/B1_fixture/` | `messages.json` / `trace.json` / `final_answer.md` | B1 个人演示（仅这三个文件，不生成其他模块产物） |
| `outputs/B2_skills/` | `<skill>_result.json` / `skill_run_log.jsonl` | 每个 Skill 最近一次 SkillResult（同名覆盖）+ 追加运行日志 |
| `outputs/B3_tools/` | `tools_schema.json` / `tool_schema_report.json` / `tool_messages.json` / `tool_call_log.jsonl` / `tool_call_stats.json` | schema、ToolMessage、执行明细、进阶调用统计（含 `attempts` / `cache_hit`） |
| `outputs/B4_llm/<case>/` | `raw_model_output.json` / `ai_message.json` / `llm_run_log.jsonl` | 原始生成、规范化 AIMessage、运行日志 |
| `outputs/B5_memory/` | `selected_memory.json` / `saved_memory.json` / `memory_log.jsonl` | 记忆查找 / 保存结果（覆盖）+ 追加日志 |

> 说明：`b1_agent_runtime.py` 的 `fixture` 模式用于完全隔离的个人演示（直接消费预设 memory / tools_schema / AIMessage / ToolMessage）；`integrated` 模式仅通过 B3/B4/B5 的公开函数编排全链路。B4 的 `mock` 模式不真实加载模型，供无 GPU / 无模型 / 模块联调时调试，不作为正式基础演示截图。
