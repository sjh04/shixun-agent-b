"""B5 整合层与 Poison Gate 评测（RQ3 / RQ4，对应 proposal 6.2 节）。

RQ3：用标注的 (old_answer, new_answer, label) 样例评测 `change_report` 的
     重复 / 补充 / 冲突 三分类准确率，输出混淆矩阵与冲突检出率。
RQ4：把标注为 投毒 / 正常 的记忆逐条过 Poison Gate（对照评测语料的全局可信记忆），
     输出拦截率 TPR 与误杀率 FPR。

--judge off 走纯规则路径（相似度阈值 + 否定词启发），--judge on 启用
Qwen judge / NLI（规则自动兜底），两种模式对照即为"规则 vs LLM 判定"消融。

用法（在 code/ 目录下）：
    python run_b5_integration_eval.py --judge off --outdir ../outputs/B5_integration_rule
    python run_b5_integration_eval.py --judge on  --outdir ../outputs/B5_integration_qwen
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import yaml

from common.io_utils import read_json, read_yaml, write_json, write_text
from common.path_utils import resolve_cli_path, resolve_from_file
from b5_memory import _change_report, _memory_paths, _poison_gate, _read_index


LABELS = ["duplicate", "supplement", "conflict"]


def _materialize_config(base_config_path: Path, judge_enabled: bool, target_path: Path) -> Path:
    config = read_yaml(base_config_path)
    memory = deepcopy(config["memory"])
    memory["root_dir"] = str(resolve_from_file(memory["root_dir"], base_config_path))
    llm = memory.setdefault("llm", {})
    if isinstance(llm.get("model_config"), str):
        llm["model_config"] = str(resolve_from_file(llm["model_config"], base_config_path))
    llm["enabled"] = judge_enabled
    memory.setdefault("integration", {})["use_llm_judge"] = judge_enabled
    gate = memory.setdefault("poison_gate", {})
    gate["enabled"] = True
    gate["use_llm_judge"] = judge_enabled
    target_path.parent.mkdir(parents=True, exist_ok=True)
    write_text(yaml.safe_dump({"memory": memory}, allow_unicode=True, sort_keys=False), target_path)
    return target_path


def _eval_integration(paths: dict, cases: list[dict]) -> dict:
    confusion = {label: {predicted: 0 for predicted in LABELS} for label in LABELS}
    records = []
    for case in cases:
        existing_markdown = f"# old\n\n## Final Answer\n\n{case['old_answer']}\n"
        report = _change_report(paths["config_path"], existing_markdown, case["new_answer"], paths["config"])
        predicted = report["change_type"]
        confusion[case["label"]][predicted] += 1
        records.append(
            {
                "case_id": case["case_id"],
                "label": case["label"],
                "predicted": predicted,
                "correct": predicted == case["label"],
                "method": report.get("method"),
                "rule_change_type": report.get("rule_change_type"),
                "similarity": report.get("similarity"),
                "judge_reason": report.get("judge_reason"),
            }
        )
    total = len(records)
    correct = sum(record["correct"] for record in records)
    conflict_total = sum(1 for case in cases if case["label"] == "conflict")
    conflict_found = confusion["conflict"]["conflict"]
    return {
        "case_count": total,
        "accuracy": correct / total if total else None,
        "conflict_recall": conflict_found / conflict_total if conflict_total else None,
        "confusion_matrix": confusion,
        "records": records,
    }


def _eval_poison(paths: dict, cases: list[dict]) -> dict:
    index = _read_index(paths["index"])
    records = []
    for case in cases:
        result = _poison_gate(paths, index, "conversation", case["answer"])
        records.append(
            {
                "case_id": case["case_id"],
                "poison": case["poison"],
                "flagged": result["flagged"],
                "correct": result["flagged"] == case["poison"],
                "method": result.get("method"),
                "matches": result.get("matches", []),
            }
        )
    poison_cases = [record for record in records if record["poison"]]
    benign_cases = [record for record in records if not record["poison"]]
    tpr = sum(record["flagged"] for record in poison_cases) / len(poison_cases) if poison_cases else None
    fpr = sum(record["flagged"] for record in benign_cases) / len(benign_cases) if benign_cases else None
    return {
        "case_count": len(records),
        "poison_count": len(poison_cases),
        "benign_count": len(benign_cases),
        "tpr_interception": tpr,
        "fpr_false_kill": fpr,
        "records": records,
    }


def _summary_markdown(judge_flag: str, integration: dict, poison: dict) -> str:
    lines = [
        f"# B5 整合层 / Poison Gate 评测（judge={judge_flag}）",
        "",
        "## RQ3 三分类（change_report）",
        "",
        f"- 样例数：{integration['case_count']}",
        f"- 准确率：{integration['accuracy']:.3f}",
        f"- 冲突检出率：{integration['conflict_recall']:.3f}",
        "",
        "混淆矩阵（行=标注，列=预测）：",
        "",
        "| 标注\\预测 | duplicate | supplement | conflict |",
        "| --- | --- | --- | --- |",
    ]
    for label in LABELS:
        row = integration["confusion_matrix"][label]
        lines.append(f"| {label} | {row['duplicate']} | {row['supplement']} | {row['conflict']} |")
    lines.extend(
        [
            "",
            "## RQ4 Poison Gate",
            "",
            f"- 投毒样例：{poison['poison_count']}，正常样例：{poison['benign_count']}",
            f"- 拦截率 TPR：{poison['tpr_interception']:.3f}",
            f"- 误杀率 FPR：{poison['fpr_false_kill']:.3f}",
            "",
        ]
    )
    return "\n".join(lines)


def run(base_config: str, integration_cases: str, poison_cases: str, outdir: str, judge_flag: str) -> dict:
    base_config_path = resolve_cli_path(base_config)
    output_dir = resolve_cli_path(outdir)
    judge_enabled = judge_flag == "on"
    config_path = _materialize_config(base_config_path, judge_enabled, output_dir / "configs" / "integration_eval.yaml")
    paths = _memory_paths(config_path)
    integration = _eval_integration(paths, read_json(resolve_cli_path(integration_cases)))
    print(f"[RQ3] accuracy={integration['accuracy']:.3f} conflict_recall={integration['conflict_recall']:.3f}")
    poison = _eval_poison(paths, read_json(resolve_cli_path(poison_cases)))
    print(f"[RQ4] TPR={poison['tpr_interception']:.3f} FPR={poison['fpr_false_kill']:.3f}")
    summary = {"judge": judge_flag, "integration": integration, "poison_gate": poison}
    write_json(summary, output_dir / "integration_eval.json")
    write_text(_summary_markdown(judge_flag, integration, poison), output_dir / "integration_eval.md")
    print(output_dir / "integration_eval.md")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate B5 integration change_report and poison gate.")
    parser.add_argument("--base_config", default="../configs/memory_eval_corpus.yaml")
    parser.add_argument("--integration_cases", default="../data/memory_eval/integration_cases.json")
    parser.add_argument("--poison_cases", default="../data/memory_eval/poison_cases.json")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--judge", choices=["on", "off"], default="off")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args.base_config, args.integration_cases, args.poison_cases, args.outdir, args.judge)
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
