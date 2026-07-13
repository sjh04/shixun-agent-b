"""B5 压缩层评测（RQ2，对应 proposal 6.2 节）。

对每条带关键点清单的超长样本，用三种方法压到同一预算：
    hard_truncate   基础版行为：text[:budget] 硬截断
    extractive      抽取式摘要（无 GPU 兜底路径）
    qwen            本地 Qwen 生成式摘要（需 --llm on，失败自动回退抽取式）

指标：
- 关键点保留率：每个关键点带 match_tokens，压缩文本中命中 ≥ 半数 token 记为保留
  （宽松口径，另报全部命中的严格口径；生成式摘要可能换说法，建议对照人工抽检）；
- 压缩比：压缩后长度 / 原文长度；
- 下游答对率（仅 --llm on）：只给压缩文本让 Qwen 回答对照题，按答案关键词判分。

用法（在 code/ 目录下）：
    python evals/run_b5_compression_eval.py --llm off --outdir ../outputs/B5_compression_rule
    python evals/run_b5_compression_eval.py --llm on  --outdir ../outputs/B5_compression_qwen
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
from b5_memory import _compress_text, _extractive_summary, _load_memory_config
from build_b5_eval_corpus import LONG_DOC_C05, LONG_DOC_C11


TEXT_SOURCES = {"LONG_DOC_C05": LONG_DOC_C05, "LONG_DOC_C11": LONG_DOC_C11}


def _materialize_config(base_config_path: Path, llm_enabled: bool, target_path: Path) -> Path:
    config = read_yaml(base_config_path)
    memory = deepcopy(config["memory"])
    memory["root_dir"] = str(resolve_from_file(memory["root_dir"], base_config_path))
    llm = memory.setdefault("llm", {})
    if isinstance(llm.get("model_config"), str):
        llm["model_config"] = str(resolve_from_file(llm["model_config"], base_config_path))
    llm["enabled"] = llm_enabled
    memory.setdefault("compression", {})["enabled"] = True
    target_path.parent.mkdir(parents=True, exist_ok=True)
    write_text(yaml.safe_dump({"memory": memory}, allow_unicode=True, sort_keys=False), target_path)
    return target_path


def _case_text(case: dict) -> str:
    if isinstance(case.get("text"), str):
        return case["text"]
    return TEXT_SOURCES[case["text_from"]]


def _keypoint_coverage(compressed: str, key_points: list[dict]) -> tuple[float, float, list[dict]]:
    lowered = compressed.casefold()
    details = []
    lenient_hits = 0
    strict_hits = 0
    for point in key_points:
        tokens = [token.casefold() for token in point["match_tokens"]]
        found = sum(1 for token in tokens if token in lowered)
        lenient = found * 2 >= len(tokens)
        strict = found == len(tokens)
        lenient_hits += lenient
        strict_hits += strict
        details.append({"id": point["id"], "found_tokens": found, "total_tokens": len(tokens), "retained_lenient": lenient, "retained_strict": strict})
    total = max(1, len(key_points))
    return lenient_hits / total, strict_hits / total, details


def _answer_with_memory(config_path: Path, config: dict, compressed: str, question: str) -> str:
    from b5_memory import _qwen_generate

    prompt = (
        "仅根据下面的记忆内容回答问题，只给结论，不要展开。\n\n"
        f"记忆：\n{compressed}\n\n问题：{question}"
    )
    return _qwen_generate(config_path, config, prompt)


def run(base_config: str, cases_path: str, outdir: str, llm_flag: str, budget: int) -> dict:
    base_config_path = resolve_cli_path(base_config)
    output_dir = resolve_cli_path(outdir)
    llm_enabled = llm_flag == "on"
    config_path = _materialize_config(base_config_path, llm_enabled, output_dir / "configs" / "compression_eval.yaml")
    _, config = _load_memory_config(config_path)
    cases = read_json(resolve_cli_path(cases_path))
    methods = ["hard_truncate", "extractive"] + (["qwen"] if llm_enabled else [])
    records = []
    for case in cases:
        text = _case_text(case)
        # 每条样本都要有压缩压力：预算取全局预算与原文 1/3 的较小者
        case_budget = min(budget, len(text) // 3)
        for method in methods:
            if method == "hard_truncate":
                compressed, actual = text[:case_budget], "hard_truncate"
            elif method == "extractive":
                compressed, actual = _extractive_summary(text, case_budget), "extractive"
            else:
                compressed, actual = _compress_text(config_path, config, text, case_budget)
            lenient, strict, details = _keypoint_coverage(compressed, case["key_points"])
            qa_results = []
            if llm_enabled:
                for item in case.get("qa", []):
                    try:
                        answer = _answer_with_memory(config_path, config, compressed, item["question"])
                    except Exception as exc:
                        answer = f"<generation failed: {exc}>"
                    correct = all(keyword.casefold() in answer.casefold() for keyword in item["answer_keywords"])
                    qa_results.append({"question": item["question"], "answer": answer[:400], "correct": correct})
            records.append(
                {
                    "case_id": case["case_id"],
                    "method": method,
                    "actual_method": actual,
                    "budget_chars": case_budget,
                    "source_chars": len(text),
                    "compressed_chars": len(compressed),
                    "compression_ratio": round(len(compressed) / max(1, len(text)), 4),
                    "keypoint_retention_lenient": round(lenient, 4),
                    "keypoint_retention_strict": round(strict, 4),
                    "keypoints": details,
                    "qa": qa_results,
                    "compressed_preview": compressed[:500],
                }
            )
            print(f"[{case['case_id']}/{method}] retention={lenient:.2f} ratio={len(compressed) / max(1, len(text)):.2f}")
    by_method = {}
    for method in methods:
        rows = [record for record in records if record["method"] == method]
        qa_flat = [item for record in rows for item in record["qa"]]
        by_method[method] = {
            "cases": len(rows),
            "mean_retention_lenient": sum(record["keypoint_retention_lenient"] for record in rows) / len(rows),
            "mean_retention_strict": sum(record["keypoint_retention_strict"] for record in rows) / len(rows),
            "mean_compression_ratio": sum(record["compression_ratio"] for record in rows) / len(rows),
            "qa_accuracy": (sum(item["correct"] for item in qa_flat) / len(qa_flat)) if qa_flat else None,
        }
    summary = {"llm": llm_flag, "budget_chars": budget, "by_method": by_method, "records": records}
    write_json(summary, output_dir / "compression_eval.json")
    lines = [
        f"# B5 压缩层评测（llm={llm_flag}，预算 {budget} 字符）",
        "",
        "| 方法 | 关键点保留率(宽松) | 关键点保留率(严格) | 压缩比 | 下游答对率 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for method, stats in by_method.items():
        qa = f"{stats['qa_accuracy']:.3f}" if stats["qa_accuracy"] is not None else "-"
        lines.append(
            f"| {method} | {stats['mean_retention_lenient']:.3f} | {stats['mean_retention_strict']:.3f} "
            f"| {stats['mean_compression_ratio']:.3f} | {qa} |"
        )
    lines.append("")
    write_text("\n".join(lines), output_dir / "compression_eval.md")
    print(output_dir / "compression_eval.md")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate B5 compression: truncate vs extractive vs Qwen summary.")
    parser.add_argument("--base_config", default="../configs/memory_eval_corpus.yaml")
    parser.add_argument("--cases", default="../data/memory_eval/compression_cases.json")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--llm", choices=["on", "off"], default="off")
    parser.add_argument("--budget", type=int, default=700)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args.base_config, args.cases, args.outdir, args.llm, args.budget)
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
