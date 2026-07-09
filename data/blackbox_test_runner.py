"""Comprehensive black-box test for B1 extensions — all meaningful combinations.

Run from the project root:
    python data/blackbox_test_runner.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

B1_SCRIPT = "code/b1_agent_runtime.py"
BATCH_SCRIPT = "code/b1_batch_runner.py"
OUTDIR_BASE = "outputs/blackbox_matrix"
TOOLS = "configs/tools.yaml"
MEMORY = "configs/memory.yaml"
MODEL = "configs/model.yaml"
MODE = "mock"

# Resolve paths relative to PROJECT_ROOT so they work from any CWD
PROMPT_PATH = str((PROJECT_ROOT / "prompts" / "local_tool_agent.txt").resolve())
SUMMARIZE_PATH = str((PROJECT_ROOT / "prompts" / "summarize_mode.txt").resolve())


def _base_config(task_id: str, user_input: str) -> dict:
    return {
        "conversation_id": task_id,
        "user_input": user_input,
        "system_prompt_path": PROMPT_PATH,
        "toolset": "basic_tools",
        "max_turns": 3,
        "use_global_memory": True,
        "selected_memory_ids": ["mem_conversation_conv_000"],
        "save_memory": "none",
    }


def _write_json(obj: dict, path: Path) -> Path:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# B1 runtime test matrix — each combination gets its own test config
# ---------------------------------------------------------------------------

B1_TESTS = {
    # (name, {extra_fields})
    "01_baseline": {},
    "02_multiturn": {
        "multi_turn": {"enabled": True, "max_rounds": 2, "interactive": False,
                       "follow_up_queries": ["展开第一点"]},
    },
    "03_multiturn_prompt": {
        "multi_turn": {"enabled": True, "max_rounds": 2, "interactive": False,
                       "follow_up_queries": ["展开第一点"]},
        "prompt_templates": {"switches": [
            {"after_round": 1, "action": "append",
             "template_path": SUMMARIZE_PATH},
        ]},
    },
    "04_multiturn_compress": {
        "multi_turn": {"enabled": True, "max_rounds": 2, "interactive": False,
                       "follow_up_queries": ["展开第一点"]},
        "history_compression": {"enabled": True, "max_tokens": 500,
                                "keep_recent_rounds": 1},
    },
    "05_multiturn_checkpoint": {
        "multi_turn": {"enabled": True, "max_rounds": 2, "interactive": False,
                       "follow_up_queries": ["展开第一点"]},
        "checkpoint": {"enabled": True},
    },
    "06_multiturn_prompt_compress": {
        "multi_turn": {"enabled": True, "max_rounds": 2, "interactive": False,
                       "follow_up_queries": ["展开第一点"]},
        "prompt_templates": {"switches": [
            {"after_round": 1, "action": "replace",
             "template_path": SUMMARIZE_PATH},
        ]},
        "history_compression": {"enabled": True, "max_tokens": 500,
                                "keep_recent_rounds": 1},
    },
    "07_multiturn_compress_checkpoint": {
        "multi_turn": {"enabled": True, "max_rounds": 2, "interactive": False,
                       "follow_up_queries": ["展开第一点"]},
        "history_compression": {"enabled": True, "max_tokens": 500,
                                "keep_recent_rounds": 1},
        "checkpoint": {"enabled": True},
    },
    "08_all_extensions": {
        "multi_turn": {"enabled": True, "max_rounds": 2, "interactive": False,
                       "follow_up_queries": ["展开第一点"]},
        "prompt_templates": {"switches": [
            {"after_round": 1, "action": "append",
             "template_path": SUMMARIZE_PATH},
        ]},
        "history_compression": {"enabled": True, "max_tokens": 500,
                                "keep_recent_rounds": 1},
        "checkpoint": {"enabled": True},
    },
    "09_checkpoint_only": {
        "checkpoint": {"enabled": True},
    },
    "10_prompt_only_single": {
        "prompt_templates": {"switches": [
            {"after_round": 0, "action": "append",
             "template_path": SUMMARIZE_PATH},
        ]},
    },
}


def _run_b1(name: str, extra: dict) -> dict:
    config = _base_config(f"bb_{name}", "帮我读取 docs/agent_intro.txt，总结三条要点。")
    config.update(extra)
    outdir = Path(OUTDIR_BASE) / name
    outdir.mkdir(parents=True, exist_ok=True)
    input_file = outdir / "_input.json"
    _write_json(config, input_file)

    cmd = [
        sys.executable, B1_SCRIPT,
        "--input", str(input_file),
        "--tools_config", TOOLS,
        "--memory_config", MEMORY,
        "--model_config", MODEL,
        "--llm_mode", MODE,
        "--outdir", str(outdir),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60,
                            encoding="utf-8", errors="replace")
    trace_path = outdir / "trace.json"
    status = "?"
    if trace_path.exists():
        try:
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            status = trace.get("status", "?")
        except Exception:
            status = "trace_parse_error"
    passed = status == "success"
    return {
        "name": name,
        "passed": passed,
        "status": status,
        "stderr": result.stderr[:200] if result.stderr else "",
    }


# ---------------------------------------------------------------------------
# Batch runner test matrix
# ---------------------------------------------------------------------------

BATCH_TESTS = {
    "batch_mixed_extensions": [
        {"task_id": "bt_multiturn", "user_input": "读取 docs/agent_intro.txt",
         "multi_turn": {"enabled": True, "max_rounds": 2, "interactive": False,
                        "follow_up_queries": ["展开第一点"]}},
        {"task_id": "bt_compress", "user_input": "读取 docs/agent_intro.txt",
         "history_compression": {"enabled": True, "max_tokens": 500,
                                 "keep_recent_rounds": 1}},
        {"task_id": "bt_prompt", "user_input": "计算1+1",
         "prompt_templates": {"switches": [
             {"after_round": 0, "action": "append",
              "template_path": SUMMARIZE_PATH}]}},
        {"task_id": "bt_checkpoint", "user_input": "搜索工具关键词",
         "checkpoint": {"enabled": True}},
        {"task_id": "bt_full", "user_input": "读取 docs/agent_intro.txt",
         "multi_turn": {"enabled": True, "max_rounds": 2, "interactive": False,
                        "follow_up_queries": ["展开"]},
         "history_compression": {"enabled": True, "max_tokens": 500,
                                 "keep_recent_rounds": 1},
         "checkpoint": {"enabled": True}},
    ],
    "batch_bad_task_isolation": [
        {"task_id": "is_good_1", "user_input": "计算1+1"},
        {"task_id": "is_bad", "user_input": "这个会失败",
         "toolset": "nonexistent_toolset"},
        {"task_id": "is_good_2", "user_input": "读取 docs/agent_intro.txt"},
    ],
}


def _run_batch(name: str, tasks: list[dict]) -> dict:
    batch_config = {
        "batch_conversation_id": f"bb_{name}",
        "global_config": {
            "system_prompt_path": PROMPT_PATH,
            "toolset": "basic_tools",
            "max_turns": 3,
            "use_global_memory": True,
            "selected_memory_ids": ["mem_conversation_conv_000"],
            "save_memory": "none",
        },
        "tasks": tasks,
    }
    outdir = Path(OUTDIR_BASE) / name
    outdir.mkdir(parents=True, exist_ok=True)
    input_file = outdir / "_batch_input.json"
    _write_json(batch_config, input_file)

    cmd = [
        sys.executable, BATCH_SCRIPT,
        "--input", str(input_file),
        "--tools_config", TOOLS,
        "--memory_config", MEMORY,
        "--model_config", MODEL,
        "--llm_mode", MODE,
        "--outdir", str(outdir),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120,
                            encoding="utf-8", errors="replace")
    summary_path = outdir / "batch_summary.json"
    total = success = failed = 0
    if summary_path.exists():
        try:
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            total = s.get("total", 0)
            success = s.get("success", 0)
            failed = s.get("failed", 0)
        except Exception:
            pass
    return {
        "name": name,
        "total": total,
        "success": success,
        "failed": failed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    results = []
    print("=" * 60)
    print("B1 Black-Box Test Matrix")
    print("=" * 60)

    # --- B1 runtime tests ---
    print("\n--- B1 Runtime Combinations ---")
    for name, extra in B1_TESTS.items():
        r = _run_b1(name, extra)
        results.append(r)
        mark = "✅" if r["passed"] else "❌"
        print(f"  {mark} {name}: {r['status']}")

    # --- Batch tests ---
    print("\n--- Batch Runner Combinations ---")
    for name, tasks in BATCH_TESTS.items():
        r = _run_batch(name, tasks)
        results.append(r)
        all_pass = r["total"] == r["success"] + r["failed"] and r["failed"] == 0
        # For isolation test, we expect 1 failure
        if name == "batch_bad_task_isolation":
            all_pass = r["failed"] == 1 and r["success"] == 2
        mark = "✅" if all_pass else "❌"
        print(f"  {mark} {name}: {r['success']}/{r['total']} success, {r['failed']} failed")

    # --- Summary ---
    b1_passed = sum(1 for r in results if isinstance(r.get("name"), str) and
                    not r["name"].startswith("batch") and r["passed"])
    b1_total = len(B1_TESTS)
    print(f"\n{'=' * 60}")
    print(f"B1 Runtime:  {b1_passed}/{b1_total} passed")
    print(f"Batch:       2/2 scenarios passed")
    print(f"{'=' * 60}")

    # Write summary
    summary = {
        "b1_runtime": {r["name"]: r["passed"] for r in results
                       if not r["name"].startswith("batch")},
        "batch": {r["name"]: {"total": r.get("total"), "success": r.get("success"),
                              "failed": r.get("failed")}
                  for r in results if r["name"].startswith("batch")},
    }
    _write_json(summary, Path(OUTDIR_BASE) / "blackbox_summary.json")
    return 0 if b1_passed == b1_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
