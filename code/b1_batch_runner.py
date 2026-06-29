from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

from b1_agent_runtime import run_agent
from common.io_utils import read_json, write_json
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path


def _validate_batch_input(payload: dict) -> dict:
    """Validate the batch input structure and fill defaults."""
    if not isinstance(payload, dict):
        raise ValueError("batch input must be a JSON object")
    if "batch_conversation_id" not in payload or not isinstance(payload["batch_conversation_id"], str):
        raise ValueError("batch_conversation_id is required and must be a string")
    if "tasks" not in payload or not isinstance(payload["tasks"], list) or len(payload["tasks"]) == 0:
        raise ValueError("tasks must be a non-empty array")

    payload.setdefault("global_config", {})
    global_config = payload["global_config"]
    if not isinstance(global_config, dict):
        raise ValueError("global_config must be an object")

    # Required fields in global_config
    required = ["system_prompt_path", "toolset", "max_turns"]
    for field in required:
        if field not in global_config:
            raise ValueError(f"global_config.{field} is required")

    global_config.setdefault("use_global_memory", False)
    global_config.setdefault("selected_memory_ids", [])
    global_config.setdefault("save_memory", "none")

    for i, task in enumerate(payload["tasks"]):
        if not isinstance(task, dict):
            raise ValueError(f"tasks[{i}] must be an object")
        if "task_id" not in task or not isinstance(task["task_id"], str):
            raise ValueError(f"tasks[{i}] missing task_id")
        if "user_input" not in task or not isinstance(task["user_input"], str):
            raise ValueError(f"tasks[{i}] missing user_input")

    return payload


def _resolve_prompt_path(path: str, base_dir: Path) -> str:
    """Resolve a system_prompt_path relative to *base_dir* so it stays valid
    when the temp runtime_input is later read from a different directory."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return str(candidate)


def _build_task_input(task: dict, global_config: dict, batch_id: str, base_dir: Path) -> dict:
    """Merge global_config with per-task overrides to build a runtime_input.

    Path values from *global_config* are resolved relative to *base_dir* so the
    temp runtime_input file can be consumed from any location.
    """
    prompt_path = task.get("system_prompt_path", global_config["system_prompt_path"])
    runtime_input = {
        "conversation_id": f"{batch_id}_{task['task_id']}",
        "user_input": task["user_input"],
        "system_prompt_path": _resolve_prompt_path(prompt_path, base_dir),
        "toolset": task.get("toolset", global_config["toolset"]),
        "max_turns": task.get("max_turns", global_config["max_turns"]),
        "use_global_memory": task.get("use_global_memory", global_config.get("use_global_memory", False)),
        "selected_memory_ids": task.get("selected_memory_ids", global_config.get("selected_memory_ids", [])),
        "save_memory": task.get("save_memory", global_config.get("save_memory", "none")),
        "execution_mode": "integrated",
    }
    # Carry forward any multi_turn config from the task or global_config
    if "multi_turn" in task:
        runtime_input["multi_turn"] = task["multi_turn"]
    elif "multi_turn" in global_config:
        runtime_input["multi_turn"] = global_config["multi_turn"]
    return runtime_input


def run_batch(
    input_path: str,
    tools_config: str,
    memory_config: str,
    model_config: str,
    outdir: str,
    llm_mode: str | None = None,
) -> dict:
    """Run a batch of Agent tasks sequentially.

    Each task runs in its own subdirectory under *outdir*.  Failures in one task
    do not stop the remaining tasks.
    """
    started = perf_counter()
    input_file = Path(input_path).resolve()
    output_dir = Path(outdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    batch = _validate_batch_input(read_json(input_file))
    global_config = batch["global_config"]
    tasks = batch["tasks"]

    print(f"Batch: {batch['batch_conversation_id']} ({len(tasks)} tasks)")
    results = []

    for idx, task in enumerate(tasks):
        task_id = task["task_id"]
        task_outdir = output_dir / task_id
        task_outdir.mkdir(parents=True, exist_ok=True)

        # Build per-task runtime input file (resolve paths relative to batch input)
        runtime_input = _build_task_input(task, global_config, batch["batch_conversation_id"], input_file.parent)
        temp_input_path = task_outdir / "_runtime_input.json"
        write_json(runtime_input, temp_input_path)

        task_start = perf_counter()
        try:
            print(f"[{idx + 1}/{len(tasks)}] {task_id}: {task['user_input'][:60]}...")
            result = run_agent(
                str(temp_input_path),
                tools_config,
                memory_config,
                model_config,
                str(task_outdir),
                llm_mode,
            )
            results.append({
                "task_id": task_id,
                "status": result["status"],
                "final_answer": result["final_answer"],
                "elapsed_ms": result["elapsed_ms"],
                "output_dir": str(task_outdir.resolve()),
                "final_answer_path": result.get("final_answer_path", ""),
            })
            print(f"  -> {result['status']} ({result['elapsed_ms']:.0f}ms)")
        except Exception as exc:
            elapsed = round((perf_counter() - task_start) * 1000, 3)
            results.append({
                "task_id": task_id,
                "status": "fatal_error",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "elapsed_ms": elapsed,
                "output_dir": str(task_outdir.resolve()),
            })
            print(f"  -> fatal_error: {type(exc).__name__}: {exc}")

    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = len(tasks) - success_count
    total_elapsed = round((perf_counter() - started) * 1000, 3)

    summary = {
        "batch_id": batch["batch_conversation_id"],
        "total": len(tasks),
        "success": success_count,
        "failed": failed_count,
        "total_elapsed_ms": total_elapsed,
        "generated_at": now_iso(),
        "results": results,
    }
    write_json(summary, output_dir / "batch_summary.json")
    print(f"\nBatch complete: {success_count}/{len(tasks)} success, {failed_count} failed ({total_elapsed:.0f}ms)")
    print(f"Summary: {output_dir / 'batch_summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multiple Agent tasks from a batch input file.")
    parser.add_argument("--input", required=True, help="Path to batch input JSON file")
    parser.add_argument("--tools_config", required=True)
    parser.add_argument("--memory_config", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--llm_mode", choices=["mock", "prompt_json"], default=None)
    parser.add_argument("--outdir", required=True, help="Output directory for all batch results")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_batch(
            str(resolve_cli_path(args.input)),
            str(resolve_cli_path(args.tools_config)),
            str(resolve_cli_path(args.memory_config)),
            str(resolve_cli_path(args.model_config)),
            str(resolve_cli_path(args.outdir)),
            args.llm_mode,
        )
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
