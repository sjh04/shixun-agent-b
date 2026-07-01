"""B2 Skill 命令行运行入口。

本文件是 B2 模块的 CLI 演示工具（NOT B3 Tool Layer）。
它提供了从命令行动态加载并执行 Skill 函数的完整流程。

主要组件：
  SKILL_MODULES  — Skill 名称到 Python 模块路径的注册表
  run_skill()    — 动态导入、参数注入、执行 Skill 并返回 SkillResult
  build_parser() — 构建 argparse 命令行参数解析器
  main()         — CLI 主流程：解析参数 → 读取输入 → 执行 → 写结果 → 记日志

使用示例：
  python b2_run_skill.py --skill calculator --input tool_inputs/input_calc.json --outdir output/
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path
from time import perf_counter

from common.io_utils import append_jsonl, read_json, write_json
from common.logging_utils import now_iso
from common.path_utils import DEFAULT_DATA_ROOT, bootstrap_project_root, resolve_cli_path
from common.schemas import make_skill_result

bootstrap_project_root()

from skills.composite import PIPELINES, run_composite_skill


SKILL_MODULES = {
    "calculator": "skills.calculator",
    "file_reader": "skills.file_reader",
    "local_file_search": "skills.local_file_search",
    "table_analyzer": "skills.table_analyzer",
    "format_converter": "skills.format_converter",
    "code_executor": "skills.code_executor",
}

PIPELINE_REGISTRY = PIPELINES


def run_skill(
    skill_name: str,
    input_data: dict,
    data_root: str | None = None,
    output_dir: str | None = None,
) -> dict:
    if skill_name not in SKILL_MODULES:
        raise ValueError(f"unknown skill: {skill_name}")

    if not isinstance(input_data, dict):
        raise ValueError("skill input must be a JSON object")

    module = importlib.import_module(SKILL_MODULES[skill_name])
    function = getattr(module, skill_name)

    kwargs = dict(input_data)

    signature = inspect.signature(function)
    if "data_root" in signature.parameters:
        kwargs["data_root"] = data_root or str(DEFAULT_DATA_ROOT)
    if "output_dir" in signature.parameters:
        kwargs["output_dir"] = output_dir

    start = perf_counter()
    try:
        output = function(** kwargs)
        status = "success"
        error = None
    except Exception as exc:
        output = None
        status = "error"
        if hasattr(exc, "as_error_dict") and callable(exc.as_error_dict):
            error = exc.as_error_dict()
        else:
            error = {"type": type(exc).__name__, "message": str(exc)}

    latency_ms = round((perf_counter() - start) * 1000, 3)

    return make_skill_result(skill_name, status, input_data, output, error, latency_ms)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one local Agent skill or composite pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--skill",
        choices=sorted(SKILL_MODULES),
        help="单个 Skill 名称",
    )
    group.add_argument(
        "--pipeline",
        choices=sorted(PIPELINE_REGISTRY),
        help="复合 Pipeline 名称（如 search_read_convert）",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="JSON 输入文件路径",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="输出目录路径",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="数据根目录（可选，不指定则使用默认值）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        input_path = resolve_cli_path(args.input)
        outdir = resolve_cli_path(args.outdir)

        input_data = read_json(input_path)

        data_root = str(resolve_cli_path(args.data_root)) if args.data_root else None

        outdir.mkdir(parents=True, exist_ok=True)

        if args.pipeline:
            result = run_composite_skill(
                args.pipeline,
                input_data,
                data_root,
                str(outdir),
            )
            exec_name = args.pipeline
        else:
            result = run_skill(args.skill, input_data, data_root, str(outdir))
            exec_name = args.skill

        result_path = outdir / f"{exec_name}_result.json"
        write_json(result, result_path)

        log_entry: dict = {
            "timestamp": now_iso(),
            "skill_name": exec_name,
            "status": result["status"],
            "result_path": str(result_path),
        }
        if args.pipeline:
            log_entry["pipeline_steps"] = len(result.get("trace", []))
            log_entry["failed_at"] = result.get("failed_at")
        else:
            log_entry["latency_ms"] = result["latency_ms"]
        append_jsonl(log_entry, outdir / "skill_run_log.jsonl")

        print(result_path)
        return 0

    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())