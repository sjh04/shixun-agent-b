"""B5 端到端记忆效用评测（RQ6，对应 proposal 6.2 节）。

一组"只有靠记忆才能答对"的任务（答案依赖 RQ1 评测语料中沉淀的项目事实），
在两个条件下让本地 Qwen 回答：
    mem_off  不注入任何记忆；
    mem_on   经 B5 完整检索管线（hybrid + chunk + 三因子 + HyDE + rerank）
             取 top-k 记忆注入后再回答。
按答案关键词判分，报告两种条件的任务成功率与差值，同时记录检索是否命中标注记忆。

说明：本脚本用"记忆注入 + 单轮回答"近似全系统链路，衡量记忆对答案正确性的
直接贡献；平均步数 / 重复提问率需要完整 B1 多轮循环，在 run_full_demo 联调中
另行演示。

用法（在 code/ 目录下，需要 GPU）：
    python evals/run_b5_e2e_eval.py --outdir ../outputs/B5_e2e
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import yaml

from _bootstrap import bootstrap_code_path

bootstrap_code_path()

from common.io_utils import read_json, read_yaml, write_json, write_text
from common.path_utils import resolve_cli_path, resolve_from_file
from b5_memory import _load_memory_config, _qwen_generate, load_memory


def _materialize_config(base_config_path: Path, target_path: Path) -> Path:
    config = read_yaml(base_config_path)
    memory = deepcopy(config["memory"])
    memory["root_dir"] = str(resolve_from_file(memory["root_dir"], base_config_path))
    llm = memory.setdefault("llm", {})
    if isinstance(llm.get("model_config"), str):
        llm["model_config"] = str(resolve_from_file(llm["model_config"], base_config_path))
    llm["enabled"] = True
    memory.setdefault("lifecycle", {})["update_access_on_load"] = False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    write_text(yaml.safe_dump({"memory": memory}, allow_unicode=True, sort_keys=False), target_path)
    return target_path


def _answer(config_path: Path, config: dict, question: str, memory_block: str | None) -> str:
    memory_part = f"以下是你此前积累的相关记忆：\n{memory_block}\n\n" if memory_block else ""
    prompt = (
        "你是本地 Agent 系统的助手，回答关于本项目的问题。\n\n"
        f"{memory_part}"
        f"用户问题：{question}\n"
        "请用中文简洁回答，只给结论。如果不知道就直说不知道。"
    )
    return _qwen_generate(config_path, config, prompt)


def run(base_config: str, tasks_path: str, outdir: str) -> dict:
    base_config_path = resolve_cli_path(base_config)
    output_dir = resolve_cli_path(outdir)
    config_path = _materialize_config(base_config_path, output_dir / "configs" / "e2e_eval.yaml")
    _, config = _load_memory_config(config_path)
    tasks = read_json(resolve_cli_path(tasks_path))
    records = []
    for task in tasks:
        result = load_memory(str(config_path), [], True, task["question"], str(output_dir / "retrieval" / task["task_id"]))
        docs = result["selected_memory_docs"]
        ranked = [doc["memory_id"] for doc in docs]
        memory_block = "\n\n".join(f"[{doc['memory_id']}] {doc['content']}" for doc in docs)
        retrieval_hit = any(memory_id in set(task["relevant_ids"]) for memory_id in ranked)
        answers = {}
        success = {}
        for condition, block in (("mem_off", None), ("mem_on", memory_block or None)):
            try:
                answer = _answer(config_path, config, task["question"], block)
            except Exception as exc:
                answer = f"<generation failed: {exc}>"
            answers[condition] = answer
            success[condition] = all(keyword.casefold() in answer.casefold() for keyword in task["answer_keywords"])
        records.append(
            {
                "task_id": task["task_id"],
                "question": task["question"],
                "answer_keywords": task["answer_keywords"],
                "retrieval_hit": retrieval_hit,
                "ranked_ids": ranked,
                "success_mem_off": success["mem_off"],
                "success_mem_on": success["mem_on"],
                "answer_mem_off": answers["mem_off"][:400],
                "answer_mem_on": answers["mem_on"][:400],
            }
        )
        print(f"[{task['task_id']}] off={'✓' if success['mem_off'] else '✗'} on={'✓' if success['mem_on'] else '✗'} retrieval_hit={retrieval_hit}")
    total = max(1, len(records))
    metrics = {
        "task_count": len(records),
        "success_rate_mem_off": sum(record["success_mem_off"] for record in records) / total,
        "success_rate_mem_on": sum(record["success_mem_on"] for record in records) / total,
        "retrieval_hit_rate": sum(record["retrieval_hit"] for record in records) / total,
    }
    metrics["delta_success_rate"] = metrics["success_rate_mem_on"] - metrics["success_rate_mem_off"]
    summary = {"metrics": metrics, "records": records}
    write_json(summary, output_dir / "e2e_eval.json")
    lines = [
        "# B5 端到端记忆效用（RQ6）",
        "",
        f"- 任务数：{metrics['task_count']}",
        f"- 无记忆成功率：{metrics['success_rate_mem_off']:.3f}",
        f"- 有记忆成功率：{metrics['success_rate_mem_on']:.3f}",
        f"- 提升：{metrics['delta_success_rate']:+.3f}",
        f"- 检索命中率（标注记忆进 top-k）：{metrics['retrieval_hit_rate']:.3f}",
        "",
        "| 任务 | 无记忆 | 有记忆 | 检索命中 |",
        "| --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record['task_id']} {record['question'][:24]} | {'✓' if record['success_mem_off'] else '✗'} "
            f"| {'✓' if record['success_mem_on'] else '✗'} | {'✓' if record['retrieval_hit'] else '✗'} |"
        )
    lines.append("")
    write_text("\n".join(lines), output_dir / "e2e_eval.md")
    print(output_dir / "e2e_eval.md")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end memory utility: answer with vs without B5 memory.")
    parser.add_argument("--base_config", default="../configs/memory_eval_corpus.yaml")
    parser.add_argument("--tasks", default="../data/memory_eval/e2e_tasks.json")
    parser.add_argument("--outdir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args.base_config, args.tasks, args.outdir)
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
