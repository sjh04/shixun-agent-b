"""B5 检索层消融矩阵跑批（RQ1，对应 proposal 6.5 节）。

每行在前一行基础上仅增加一个组件，统一在标注语料上报 Hit@k / MRR / nDCG@5 / 延迟：

    B0          基础版：不启用检索，按"全局记忆 + 指定 id"顺序返回
    KW          纯关键词（BM25，整文档）
    VEC         纯向量（整文档）
    RRF         BM25 + 向量 + RRF 融合（整文档）
    RRF_CHUNK   + chunk 级召回
    HYDE        + HyDE 查询改写（需 --llm on，否则自动回退原 query）
    THREE_F     + 三因子重排（相关度 × 时近性 × 重要性）
    FULL        + LLM rerank（需 --llm on，否则保持原序）

用法（在 code/ 目录下）：
    python run_b5_ablation.py --llm off --outdir ../outputs/B5_ablation_hashing
    python run_b5_ablation.py --llm on  --outdir ../outputs/B5_ablation_qwen

每行实际生效的配置会落盘到 <outdir>/configs/<row>.yaml，逐 query 结果在
<outdir>/<row>/，汇总表在 <outdir>/ablation_summary.{json,md}。
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import yaml

from common.io_utils import read_json, read_yaml, write_json, write_text
from common.path_utils import resolve_cli_path, resolve_from_file
from evaluate_b5_memory import evaluate


MATRIX: list[tuple[str, dict]] = [
    ("B0", {"mode": "none"}),
    ("KW", {"mode": "keyword", "use_chunk": False, "use_three_factor": False, "use_hyde": False, "use_rerank": False}),
    ("VEC", {"mode": "vector", "use_chunk": False, "use_three_factor": False, "use_hyde": False, "use_rerank": False}),
    ("RRF", {"mode": "hybrid", "use_chunk": False, "use_three_factor": False, "use_hyde": False, "use_rerank": False}),
    ("RRF_CHUNK", {"mode": "hybrid", "use_chunk": True, "use_three_factor": False, "use_hyde": False, "use_rerank": False}),
    ("HYDE", {"mode": "hybrid", "use_chunk": True, "use_three_factor": False, "use_hyde": True, "use_rerank": False}),
    ("THREE_F", {"mode": "hybrid", "use_chunk": True, "use_three_factor": True, "use_hyde": True, "use_rerank": False}),
    ("FULL", {"mode": "hybrid", "use_chunk": True, "use_three_factor": True, "use_hyde": True, "use_rerank": True}),
]


def _materialize_config(base_config_path: Path, row_overrides: dict, llm_enabled: bool, target_path: Path) -> None:
    config = read_yaml(base_config_path)
    if not isinstance(config, dict) or not isinstance(config.get("memory"), dict):
        raise ValueError("base config must define a memory object")
    memory = deepcopy(config["memory"])
    # 落盘到 outdir 后相对路径基准会变，全部转成绝对路径
    memory["root_dir"] = str(resolve_from_file(memory["root_dir"], base_config_path))
    llm = memory.setdefault("llm", {})
    if isinstance(llm.get("model_config"), str):
        llm["model_config"] = str(resolve_from_file(llm["model_config"], base_config_path))
    llm["enabled"] = llm_enabled
    retrieval = memory.setdefault("retrieval", {})
    retrieval.update(row_overrides)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    write_text(yaml.safe_dump({"memory": memory}, allow_unicode=True, sort_keys=False), target_path)


def _probe_breakdown(records: list[dict], probe_by_query: dict[str, str]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        probe = record.get("probe") or probe_by_query.get(record.get("query", ""))
        if not probe:
            continue
        grouped.setdefault(probe, []).append(record)
    breakdown = {}
    for probe, items in sorted(grouped.items()):
        total = len(items)
        breakdown[probe] = {
            "count": total,
            "hit@1": sum(item["hit@1"] for item in items) / total,
            "hit@3": sum(item["hit@3"] for item in items) / total,
            "mrr": sum(item["reciprocal_rank"] for item in items) / total,
        }
    return breakdown


def _summary_markdown(rows: list[dict], llm_flag: str) -> str:
    lines = [
        f"# B5 检索层消融矩阵（llm={llm_flag}）",
        "",
        "| 配置 | Hit@1 | Hit@3 | Hit@5 | MRR | nDCG@5 | 平均延迟(ms) | HyDE | rerank | 向量后端 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        metrics = row["metrics"]
        latency = metrics.get("mean_latency_ms")
        lines.append(
            "| {name} | {h1:.3f} | {h3:.3f} | {h5:.3f} | {mrr:.3f} | {ndcg:.3f} | {lat} | {hyde} | {rerank} | {backend} |".format(
                name=row["name"],
                h1=metrics["hit@1"],
                h3=metrics["hit@3"],
                h5=metrics["hit@5"],
                mrr=metrics["mrr"],
                ndcg=metrics["ndcg@5"],
                lat=f"{latency:.1f}" if isinstance(latency, (int, float)) else "-",
                hyde=row["diagnostics"].get("hyde", "-"),
                rerank=row["diagnostics"].get("rerank", "-"),
                backend=row["diagnostics"].get("vector_backend") or "-",
            )
        )
    probes = sorted({probe for row in rows for probe in (row.get("by_probe") or {})})
    if probes:
        lines.extend(
            [
                "",
                "## 分探针类型 Hit@1（每个组件在其针对的查询类型上证明价值）",
                "",
                "| 配置 | " + " | ".join(probes) + " |",
                "| --- |" + " --- |" * len(probes),
            ]
        )
        for row in rows:
            by_probe = row.get("by_probe") or {}
            cells = []
            for probe in probes:
                item = by_probe.get(probe)
                cells.append(f"{item['hit@1']:.2f} (n={item['count']})" if item else "-")
            lines.append(f"| {row['name']} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def run(base_config: str, queries: str, outdir: str, llm_flag: str, only: list[str] | None, summarize_only: bool = False) -> dict:
    base_config_path = resolve_cli_path(base_config)
    queries_path = resolve_cli_path(queries)
    output_dir = resolve_cli_path(outdir)
    llm_enabled = llm_flag == "on"
    probe_by_query = {
        item["query"]: item["probe"]
        for item in read_json(queries_path)
        if isinstance(item, dict) and isinstance(item.get("probe"), str)
    }
    rows = []
    for name, overrides in MATRIX:
        if only and name not in only:
            continue
        report_path = output_dir / name.lower() / "b5_retrieval_eval.json"
        if summarize_only:
            if not report_path.exists():
                print(f"[{name}] skipped: {report_path} not found")
                continue
            report = read_json(report_path)
        else:
            config_path = output_dir / "configs" / f"{name.lower()}.yaml"
            _materialize_config(base_config_path, overrides, llm_enabled, config_path)
            report = evaluate(str(config_path), str(queries_path), str(output_dir / name.lower()))
        first_retrieval = next(
            (record["retrieval"] for record in report["records"] if isinstance(record.get("retrieval"), dict)),
            {},
        )
        diagnostics = {key: first_retrieval.get(key) for key in ("mode", "hyde", "rerank", "vector_backend", "use_chunk", "use_three_factor")}
        rows.append(
            {
                "name": name,
                "overrides": overrides,
                "metrics": report["metrics"],
                "diagnostics": diagnostics,
                "by_probe": _probe_breakdown(report["records"], probe_by_query),
            }
        )
        print(f"[{name}] hit@1={report['metrics']['hit@1']:.3f} hit@3={report['metrics']['hit@3']:.3f} mrr={report['metrics']['mrr']:.3f}")
    summary = {"llm": llm_flag, "base_config": str(base_config_path), "queries": str(queries_path), "rows": rows}
    write_json(summary, output_dir / "ablation_summary.json")
    write_text(_summary_markdown(rows, llm_flag), output_dir / "ablation_summary.md")
    print(output_dir / "ablation_summary.md")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the B5 retrieval ablation matrix.")
    parser.add_argument("--base_config", default="../configs/memory_eval_corpus.yaml")
    parser.add_argument("--queries", default="../data/memory_eval/corpus_queries.json")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--llm", choices=["on", "off"], default="off")
    parser.add_argument("--only", nargs="*", help="仅运行指定行，如 --only B0 FULL")
    parser.add_argument("--summarize_only", action="store_true", help="不重跑检索，仅由已有逐行结果重新生成汇总表")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args.base_config, args.queries, args.outdir, args.llm, args.only, args.summarize_only)
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
