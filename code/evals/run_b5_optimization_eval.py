"""B5 优化回归评测。

覆盖本轮 save/load 优化：
1. retrieval_summary / retrieval_terms 对文件名、错误码、命令和工具名检索的增益；
2. query embedding cache 的重复查询命中；
3. save-time prewarm 后首次 load 不再重算 chunk vectors；
4. memory 更新后 source_hash 变化，旧 chunk/vector 不会被误用。

用法（在 code/ 目录下）：
    python evals/run_b5_optimization_eval.py --outdir ../outputs/B5_optimization
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from copy import deepcopy
from pathlib import Path

import yaml

from _bootstrap import bootstrap_code_path

bootstrap_code_path()

from b5_memory import load_memory, save_memory
from common.io_utils import read_json, write_json, write_text
from common.path_utils import resolve_cli_path


CASES = [
    {
        "conversation_id": "meta_yaml",
        "answer": "已定位为运行环境依赖缺失，处理方式是补齐缺少的包并复查配置。",
        "messages": [{"role": "user", "content": "环境启动失败，日志里有 yaml 相关报错。"}],
        "trace": {
            "tool_names": ["exec_command"],
            "touched_paths": ["agent/configs/model.yaml"],
            "error": "ModuleNotFoundError",
            "command": "python3 -m pip install PyYAML",
        },
        "query": "ModuleNotFoundError yaml 配置文件怎么修",
        "relevant_id": "mem_conversation_meta_yaml",
        "probe": "error_file",
    },
    {
        "conversation_id": "meta_cuda",
        "answer": "已确认是资源不足导致的加载失败，处理方式是降低精度并限制模型占用。",
        "messages": [{"role": "user", "content": "模型加载过程中显存不够。"}],
        "trace": {"tool_names": ["read_json"], "error": "CUDA out of memory", "settings": ["bfloat16", "device_map", "max_memory"]},
        "query": "OOM device_map max_memory",
        "relevant_id": "mem_conversation_meta_cuda",
        "probe": "error_config",
    },
    {
        "conversation_id": "meta_json",
        "answer": "已确认是结构化输出前面混入额外文本，处理方式是先截取有效对象再解析。",
        "messages": [{"role": "user", "content": "解析模型输出失败。"}],
        "trace": {"tool_names": ["file_reader"], "error": "JSONDecodeError"},
        "query": "Expecting value json.loads",
        "relevant_id": "mem_conversation_meta_json",
        "probe": "error_code",
    },
    {
        "conversation_id": "meta_command",
        "answer": "已确认完整检索评测需要使用专用环境运行消融脚本，并只选择完整配置行。",
        "messages": [{"role": "user", "content": "怎么跑 Qwen 检索评测？"}],
        "trace": {"tool_names": ["exec_command"], "command": "/home/ubuntu/miniconda3/envs/agent/bin/python evals/run_b5_ablation.py --llm on --only FULL"},
        "query": "run_b5_ablation llm on FULL 命令",
        "relevant_id": "mem_conversation_meta_command",
        "probe": "command",
    },
    {
        "conversation_id": "meta_tool",
        "answer": "已确认本地资料读取和路径搜索需要分别走两个不同的工具入口。",
        "messages": [{"role": "user", "content": "读文件和搜文件分别用哪个工具？"}],
        "trace": {"tool_names": ["file_reader", "local_file_search"]},
        "query": "local_file_search file_reader 工具",
        "relevant_id": "mem_conversation_meta_tool",
        "probe": "tool_name",
    },
]


BASE_MEMORY_CONFIG = {
    "root_dir": "memory",
    "global_memory_dir": "global",
    "conversation_memory_dir": "conversations",
    "index_path": "memory_index.json",
    "max_memory_chars": 6000,
    "retrieval": {
        "mode": "hybrid",
        "top_k": 3,
        "candidate_limit": 20,
        "chunk_chars": 500,
        "chunk_overlap": 80,
        "rrf_k": 60,
        "recency_half_life_days": 30,
        "include_all_conversations": True,
        "include_global_when_query": True,
        "use_chunk": True,
        "use_three_factor": True,
        "three_factor_weights": {"relevance": 0.8, "recency": 0.1, "importance": 0.1},
        "three_factor_top_m": 10,
        "use_hyde": False,
        "use_rerank": False,
        "vector_backend": "hashing",
        "hashing_dim": 256,
        "cache_embeddings": True,
        "retrieval_cache_path": "memory_retrieval_cache.sqlite3",
    },
    "compression": {"enabled": False, "summary_chars": 500, "summary_sentences": 5, "min_chars_for_summary": 480},
    "integration": {"enabled": True, "duplicate_similarity": 0.92, "conflict_similarity": 0.28, "use_llm_judge": False},
    "poison_gate": {"enabled": False, "action": "flag", "trusted_memory_types": ["global"], "conflict_threshold": 0.32, "use_llm_judge": False},
    "lifecycle": {
        "max_memories": 50,
        "protect_global": True,
        "importance_default": 5,
        "update_access_on_load": False,
        "reflect_enabled": False,
        "reflect_every_n": 10,
        "reflect_min_cluster_size": 3,
        "reflect_similarity_threshold": 0.35,
    },
    "llm": {"enabled": False, "model_config": "model.yaml", "mode": "prompt_json", "max_new_tokens": 64},
}


def _write_config(root: Path, metadata_enabled: bool) -> Path:
    memory = deepcopy(BASE_MEMORY_CONFIG)
    config_path = root / ("memory_meta_on.yaml" if metadata_enabled else "memory_meta_off.yaml")
    write_text(yaml.safe_dump({"memory": memory}, allow_unicode=True, sort_keys=False), config_path)
    return config_path


def _input_files(root: Path, case: dict, answer: str | None = None) -> tuple[Path, Path, Path]:
    inputs = root / "inputs" / case["conversation_id"]
    inputs.mkdir(parents=True, exist_ok=True)
    messages = inputs / "messages.json"
    trace = inputs / "trace.json"
    answer_path = inputs / "answer.md"
    write_text(json.dumps(case["messages"], ensure_ascii=False), messages)
    write_text(json.dumps(case["trace"], ensure_ascii=False), trace)
    write_text(answer if answer is not None else case["answer"], answer_path)
    return messages, trace, answer_path


def _prepare_corpus(root: Path, metadata_enabled: bool) -> Path:
    (root / "memory" / "global").mkdir(parents=True, exist_ok=True)
    (root / "memory" / "conversations").mkdir(parents=True, exist_ok=True)
    config_path = _write_config(root, metadata_enabled)
    for case in CASES:
        messages, trace, answer = _input_files(root, case)
        save_memory(
            str(config_path),
            case["conversation_id"],
            "conversation",
            str(messages),
            str(trace),
            str(answer),
            str(root / "save_outputs" / case["conversation_id"]),
        )
    if not metadata_enabled:
        index_path = root / "memory" / "memory_index.json"
        index = read_json(index_path)
        for item in index.values():
            item.pop("retrieval_summary", None)
            item.pop("retrieval_terms", None)
        write_json(index, index_path)
        cache_path = root / "memory" / "memory_retrieval_cache.sqlite3"
        if cache_path.exists():
            cache_path.unlink()
    return config_path


def _evaluate_queries(config_path: Path, root: Path) -> tuple[dict, list[dict]]:
    records = []
    for case in CASES:
        result = load_memory(str(config_path), [], True, case["query"], str(root / "loads" / case["conversation_id"]))
        ranked = [doc["memory_id"] for doc in result["selected_memory_docs"]]
        first_rank = ranked.index(case["relevant_id"]) + 1 if case["relevant_id"] in ranked else None
        records.append(
            {
                "query": case["query"],
                "probe": case["probe"],
                "relevant_id": case["relevant_id"],
                "ranked_ids": ranked,
                "first_rank": first_rank,
                "hit@1": first_rank == 1,
                "hit@3": bool(first_rank and first_rank <= 3),
                "retrieval": result["retrieval"],
            }
        )
    total = max(1, len(records))
    metrics = {
        "query_count": len(records),
        "hit@1": sum(record["hit@1"] for record in records) / total,
        "hit@3": sum(record["hit@3"] for record in records) / total,
        "mrr": sum(0.0 if record["first_rank"] is None else 1.0 / record["first_rank"] for record in records) / total,
        "mean_latency_ms": sum(record["retrieval"]["latency_ms"] for record in records) / total,
    }
    return metrics, records


def _sqlite_counts(root: Path) -> dict:
    db = root / "memory" / "memory_retrieval_cache.sqlite3"
    connection = sqlite3.connect(db)
    try:
        return {
            "documents": connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            "chunks": connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "vectors": connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0],
            "query_vectors": connection.execute("SELECT COUNT(*) FROM query_vectors").fetchone()[0],
        }
    finally:
        connection.close()


def _eval_metadata(output_dir: Path) -> dict:
    roots = {"metadata_on": output_dir / "metadata_on", "metadata_off": output_dir / "metadata_off"}
    results = {}
    for name, root in roots.items():
        config_path = _prepare_corpus(root, name == "metadata_on")
        metrics, records = _evaluate_queries(config_path, root)
        results[name] = {"metrics": metrics, "records": records, "sqlite_counts": _sqlite_counts(root)}
    return results


def _eval_cache_and_update(output_dir: Path) -> dict:
    root = output_dir / "cache_update"
    config_path = _prepare_corpus(root, True)
    case = CASES[0]
    first = load_memory(str(config_path), [], True, case["query"], str(root / "cache_first"))
    second = load_memory(str(config_path), [], True, case["query"], str(root / "cache_second"))
    before_counts = _sqlite_counts(root)
    messages, trace, answer = _input_files(
        root,
        case,
        "更新后的结论：PyYAML 已安装时无需 pip；如果报 ImportError，请检查 agent/configs/runtime.yaml。",
    )
    save_memory(
        str(config_path),
        case["conversation_id"],
        "conversation",
        str(messages),
        str(trace),
        str(answer),
        str(root / "save_update"),
    )
    updated = load_memory(str(config_path), [], True, "runtime.yaml ImportError PyYAML", str(root / "load_updated"))
    after_counts = _sqlite_counts(root)
    index = read_json(root / "memory" / "memory_index.json")
    updated_meta = index[case["relevant_id"]]
    return {
        "first_load_query_cache": first["retrieval"]["query_embedding_cache"],
        "second_load_query_cache": second["retrieval"]["query_embedding_cache"],
        "first_load_embedding_cache": first["retrieval"]["embedding_cache"],
        "prewarm_after_save": read_json(root / "save_outputs" / case["conversation_id"] / "saved_memory.json")["retrieval_cache"],
        "before_update_counts": before_counts,
        "after_update_counts": after_counts,
        "updated_ranked_ids": [doc["memory_id"] for doc in updated["selected_memory_docs"]],
        "updated_retrieval_summary": updated_meta.get("retrieval_summary"),
        "updated_files": updated_meta.get("retrieval_terms", {}).get("files", []),
    }


def _summary_markdown(metadata: dict, cache_update: dict) -> str:
    on = metadata["metadata_on"]["metrics"]
    off = metadata["metadata_off"]["metrics"]
    return "\n".join(
        [
            "# B5 优化回归评测",
            "",
            "## Metadata Ablation",
            "",
            "| 设置 | Hit@1 | Hit@3 | MRR | 平均延迟(ms) |",
            "| --- | --- | --- | --- | --- |",
            f"| metadata off | {off['hit@1']:.3f} | {off['hit@3']:.3f} | {off['mrr']:.3f} | {off['mean_latency_ms']:.1f} |",
            f"| metadata on | {on['hit@1']:.3f} | {on['hit@3']:.3f} | {on['mrr']:.3f} | {on['mean_latency_ms']:.1f} |",
            "",
            "## Cache / Update Invalidation",
            "",
            f"- 第二次相同 query cache：{cache_update['second_load_query_cache']}",
            f"- save-time prewarm：{cache_update['prewarm_after_save']}",
            f"- 更新前 SQLite 计数：{cache_update['before_update_counts']}",
            f"- 更新后 SQLite 计数：{cache_update['after_update_counts']}",
            f"- 更新后检索摘要：{cache_update['updated_retrieval_summary']}",
            f"- 更新后文件 signals：{cache_update['updated_files']}",
            "",
        ]
    )


def run(outdir: str) -> dict:
    output_dir = resolve_cli_path(outdir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _eval_metadata(output_dir)
    print(
        "[metadata] off hit@1={:.3f} on hit@1={:.3f}".format(
            metadata["metadata_off"]["metrics"]["hit@1"],
            metadata["metadata_on"]["metrics"]["hit@1"],
        )
    )
    cache_update = _eval_cache_and_update(output_dir)
    print(
        "[cache] second_query_hits={} updated_files={}".format(
            cache_update["second_load_query_cache"]["hits"],
            cache_update["updated_files"],
        )
    )
    report = {"status": "success", "metadata_ablation": metadata, "cache_update": cache_update}
    write_json(report, output_dir / "optimization_eval.json")
    write_text(_summary_markdown(metadata, cache_update), output_dir / "optimization_eval.md")
    print(output_dir / "optimization_eval.md")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run B5 optimization regression tests.")
    parser.add_argument("--outdir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args.outdir)
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
