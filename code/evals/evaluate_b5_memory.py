from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from _bootstrap import bootstrap_code_path

bootstrap_code_path()

from b5_memory import load_memory
from common.io_utils import read_json, write_json
from common.path_utils import resolve_cli_path


def _validate_queries(payload: object) -> list[dict]:
    if not isinstance(payload, list):
        raise ValueError("query file must contain a JSON array")
    queries = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"query item {index} must be an object")
        query = item.get("query")
        relevant_ids = item.get("relevant_ids")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"query item {index} missing query")
        if not isinstance(relevant_ids, list) or not all(isinstance(value, str) for value in relevant_ids):
            raise ValueError(f"query item {index} relevant_ids must be a list of strings")
        probe = item.get("probe")
        queries.append({"query": query, "relevant_ids": relevant_ids, "probe": probe if isinstance(probe, str) else None})
    return queries


def _first_relevant_rank(ranked_ids: list[str], relevant: set[str]) -> int | None:
    for rank, memory_id in enumerate(ranked_ids, 1):
        if memory_id in relevant:
            return rank
    return None


def _ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int = 5) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, memory_id in enumerate(ranked_ids[:k], 1)
        if memory_id in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate(config_path: str, queries_path: str, outdir: str) -> dict:
    queries = _validate_queries(read_json(queries_path))
    output_dir = Path(outdir)
    records = []
    for index, item in enumerate(queries, 1):
        case_dir = output_dir / f"query_{index:03d}"
        result = load_memory(
            config_path,
            selected_memory_ids=[],
            use_global_memory=True,
            query=item["query"],
            outdir=str(case_dir),
        )
        ranked_ids = [doc["memory_id"] for doc in result.get("selected_memory_docs", [])]
        relevant = set(item["relevant_ids"])
        first_rank = _first_relevant_rank(ranked_ids, relevant)
        records.append(
            {
                "query": item["query"],
                "probe": item.get("probe"),
                "relevant_ids": item["relevant_ids"],
                "ranked_ids": ranked_ids,
                "first_relevant_rank": first_rank,
                "hit@1": bool(first_rank and first_rank <= 1),
                "hit@3": bool(first_rank and first_rank <= 3),
                "hit@5": bool(first_rank and first_rank <= 5),
                "reciprocal_rank": 0.0 if first_rank is None else 1.0 / first_rank,
                "ndcg@5": _ndcg_at_k(ranked_ids, relevant),
                "retrieval": result.get("retrieval"),
                "errors": result.get("errors", []),
            }
        )
    total = max(1, len(records))
    latencies = [
        record["retrieval"]["latency_ms"]
        for record in records
        if isinstance(record.get("retrieval"), dict) and isinstance(record["retrieval"].get("latency_ms"), (int, float))
    ]
    metrics = {
        "query_count": len(records),
        "hit@1": sum(record["hit@1"] for record in records) / total,
        "hit@3": sum(record["hit@3"] for record in records) / total,
        "hit@5": sum(record["hit@5"] for record in records) / total,
        "mrr": sum(record["reciprocal_rank"] for record in records) / total,
        "ndcg@5": sum(record["ndcg@5"] for record in records) / total,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else None,
    }
    report = {"status": "success", "metrics": metrics, "records": records}
    write_json(report, output_dir / "b5_retrieval_eval.json")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate B5 memory retrieval with Hit@k and MRR.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--outdir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outdir = resolve_cli_path(args.outdir)
        evaluate(str(resolve_cli_path(args.config)), str(resolve_cli_path(args.queries)), str(outdir))
        print(outdir / "b5_retrieval_eval.json")
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
