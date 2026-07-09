from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from common.io_utils import append_jsonl, read_json, read_text, read_yaml, write_json, write_text
from common.logging_utils import now_iso
from common.path_utils import resolve_cli_path, resolve_from_file
from common.schemas import validate_ai_message


# Tools that are always safe to execute without user confirmation
SAFE_TOOLS: set[str] = {
    "file_reader",
    "calculator",
    "local_file_search",
    "table_analyzer",
    "format_converter",
}


def _validate_runtime_input(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("runtime_input.json must contain an object")
    execution_mode = payload.setdefault("execution_mode", "integrated")
    if execution_mode not in {"integrated", "fixture"}:
        raise ValueError("execution_mode must be integrated or fixture")
    required = ["conversation_id", "user_input", "system_prompt_path", "toolset", "max_turns", "save_memory"]
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"runtime input missing: {', '.join(missing)}")
    if not isinstance(payload["conversation_id"], str) or not payload["conversation_id"]:
        raise ValueError("conversation_id must be a non-empty string")
    if not isinstance(payload["user_input"], str) or not payload["user_input"].strip():
        raise ValueError("user_input must be a non-empty string")
    if not isinstance(payload["max_turns"], int) or isinstance(payload["max_turns"], bool) or payload["max_turns"] < 1:
        raise ValueError("max_turns must be a positive integer")
    if payload["save_memory"] not in {"none", "conversation", "global"}:
        raise ValueError("save_memory must be none, conversation, or global")
    if execution_mode == "fixture":
        fixtures = payload.get("fixtures")
        if not isinstance(fixtures, dict):
            raise ValueError("fixture mode requires a fixtures object")
        required_fixtures = [
            "selected_memory_path",
            "tools_schema_path",
            "ai_messages_path",
            "tool_messages_path",
        ]
        missing_fixtures = [field for field in required_fixtures if not isinstance(fixtures.get(field), str)]
        if missing_fixtures:
            raise ValueError(f"fixtures missing paths: {', '.join(missing_fixtures)}")
        if payload["save_memory"] != "none":
            raise ValueError("fixture mode requires save_memory=none")
    else:
        selected_ids = payload.setdefault("selected_memory_ids", [])
        if not isinstance(selected_ids, list) or not all(isinstance(item, str) for item in selected_ids):
            raise ValueError("selected_memory_ids must be a list of strings")
        payload.setdefault("use_global_memory", False)
        if not isinstance(payload["use_global_memory"], bool):
            raise ValueError("use_global_memory must be boolean")
    # --- multi_turn validation ---
    multi_turn = payload.setdefault("multi_turn", {})
    if not isinstance(multi_turn, dict):
        raise ValueError("multi_turn must be an object")
    if multi_turn.get("enabled"):
        max_rounds = multi_turn.get("max_rounds", 1)
        if not isinstance(max_rounds, int) or max_rounds < 1:
            raise ValueError("multi_turn.max_rounds must be a positive integer")
        if not multi_turn.get("interactive") and not isinstance(multi_turn.get("follow_up_queries"), list):
            raise ValueError("multi_turn requires interactive=true or follow_up_queries list")
        if multi_turn.get("interactive") and execution_mode == "fixture":
            raise ValueError("multi_turn interactive mode is not supported in fixture mode")
    # --- prompt_templates validation ---
    prompt_templates = payload.setdefault("prompt_templates", {})
    if not isinstance(prompt_templates, dict):
        raise ValueError("prompt_templates must be an object")
    switches = prompt_templates.get("switches", [])
    if not isinstance(switches, list):
        raise ValueError("prompt_templates.switches must be a list")
    for i, rule in enumerate(switches):
        if not isinstance(rule, dict):
            raise ValueError(f"prompt_templates.switches[{i}] must be an object")
        if "template_path" not in rule:
            raise ValueError(f"prompt_templates.switches[{i}] missing template_path")
        if "action" in rule and rule["action"] not in ("replace", "append"):
            raise ValueError(f"prompt_templates.switches[{i}].action must be replace or append")
    # --- history_compression validation ---
    hc = payload.setdefault("history_compression", {})
    if not isinstance(hc, dict):
        raise ValueError("history_compression must be an object")
    if hc.get("enabled"):
        if not isinstance(hc.get("max_tokens", 3500), int) or hc["max_tokens"] < 100:
            raise ValueError("history_compression.max_tokens must be an integer >= 100")
        if not isinstance(hc.get("keep_recent_rounds", 3), int) or hc["keep_recent_rounds"] < 1:
            raise ValueError("history_compression.keep_recent_rounds must be a positive integer")
    # --- checkpoint validation ---
    ckpt = payload.setdefault("checkpoint", {})
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint must be an object")
    if ckpt.get("enabled") and not isinstance(ckpt.get("enabled"), bool):
        raise ValueError("checkpoint.enabled must be boolean")
    # --- confirm_mode validation ---
    payload.setdefault("confirm_mode", False)
    if not isinstance(payload.get("confirm_mode"), bool):
        raise ValueError("confirm_mode must be boolean")
    safe_tools = payload.setdefault("safe_tools", list(SAFE_TOOLS))
    if not isinstance(safe_tools, list) or not all(isinstance(t, str) for t in safe_tools):
        raise ValueError("safe_tools must be a list of strings")
    # --- plan_mode validation ---
    payload.setdefault("plan_mode", False)
    if not isinstance(payload.get("plan_mode"), bool):
        raise ValueError("plan_mode must be boolean")
    return payload


def _memory_context(selected_memory: dict) -> str:
    sections = []
    for document in selected_memory.get("selected_memory_docs", []):
        sections.append(
            f'<memory id="{document["memory_id"]}" type="{document["memory_type"]}">\n'
            f'{document["content"].strip()}\n</memory>'
        )
    return "\n\n".join(sections)


def _default_llm_mode(model_config: Path) -> str:
    config = read_yaml(model_config)
    return config.get("runtime", {}).get("default_mode", "mock")


def generate_ai_message(*args, **kwargs) -> dict:
    """Lazy B4 proxy retained as the integrated-mode injection point."""
    from b4_local_agent_llm import generate_ai_message as b4_generate_ai_message

    return b4_generate_ai_message(*args, **kwargs)


def _load_fixture_inputs(input_file: Path, runtime: dict) -> dict:
    fixtures = runtime["fixtures"]
    selected_memory = read_json(resolve_from_file(fixtures["selected_memory_path"], input_file))
    tools_schema = read_json(resolve_from_file(fixtures["tools_schema_path"], input_file))
    ai_messages = read_json(resolve_from_file(fixtures["ai_messages_path"], input_file))
    tool_messages = read_json(resolve_from_file(fixtures["tool_messages_path"], input_file))
    if not isinstance(selected_memory, dict):
        raise ValueError("preset memory must be a JSON object")
    if not isinstance(tools_schema, list):
        raise ValueError("preset tools_schema must be a JSON array")
    if not isinstance(ai_messages, list) or not ai_messages:
        raise ValueError("preset AI messages must be a non-empty JSON array")
    if not isinstance(tool_messages, dict):
        raise ValueError("preset ToolMessages must be an object keyed by tool_call_id")
    for message in ai_messages:
        validate_ai_message(message)
    return {
        "selected_memory": selected_memory,
        "tools_schema": tools_schema,
        "ai_messages": ai_messages,
        "tool_messages": tool_messages,
    }


def _fixture_tool_messages(tool_calls: list[dict], preset_messages: dict) -> list[dict]:
    results = []
    for call in tool_calls:
        call_id = call.get("id")
        message = deepcopy(preset_messages.get(call_id))
        if not isinstance(message, dict):
            raise ValueError(f"fixture ToolMessage does not exist for tool_call_id: {call_id}")
        if message.get("role") != "tool" or message.get("tool_call_id") != call_id:
            raise ValueError(f"invalid fixture ToolMessage for tool_call_id: {call_id}")
        if message.get("name") != call.get("name"):
            raise ValueError(f"fixture ToolMessage name does not match call: {call_id}")
        results.append(message)
    return results


def _agent_loop(
    messages: list[dict],
    tools_schema: list[dict],
    runtime: dict,
    execution_mode: str,
    fixture_data: dict | None,
    tools_file: Path | None,
    model_file: Path | None,
    output_dir: Path,
    mode: str,
    selected_memory: dict,
    llm_call_start: int,
    execute_tool_calls: "callable | None" = None,
    resume_stage: str | None = None,
    pending_tool_calls: list[dict] | None = None,
) -> dict:
    """Execute one complete Agent tool-calling loop for the current user message.

    checkpoint-enabled: when ``runtime`` contains ``checkpoint.enabled: true``,
    snapshots are persisted at ``after_llm`` and ``after_tool_{i}`` stages.
    On resume the caller passes *resume_stage* and *pending_tool_calls* to
    skip already-completed work.
    """
    tool_rounds = 0
    llm_calls = 0
    turns: list[dict] = []
    all_tool_messages: list[dict] = []
    final_answer = ""
    status = "success"
    terminal_error = None
    warnings: list[str] = []
    ckpt_enabled = runtime.get("checkpoint", {}).get("enabled", False)
    if selected_memory.get("status") in {"partial", "error"}:
        warnings.append("memory selection completed with errors")

    while True:
        # ── checkpoint: skip LLM call if resuming from after_llm ──
        turn_start = perf_counter()
        if resume_stage == "after_llm" and pending_tool_calls:
            # The checkpoint already contains this assistant message — don't re-append
            ai_message = {"role": "assistant", "content": "", "tool_calls": pending_tool_calls}
            llm_status = "success"
            llm_error = None
            resume_stage = None  # consume the resume stage
            llm_calls += 1  # count the resumed LLM call
            is_resume_message = True
        else:
            is_resume_message = False
            llm_calls += 1
            if execution_mode == "fixture":
                if llm_calls > len(fixture_data["ai_messages"]):
                    raise ValueError("fixture AIMessage sequence ended before a final answer")
                ai_message = deepcopy(fixture_data["ai_messages"][llm_calls - 1])
                llm_status = "success"
                llm_error = None
            else:
                llm_result = generate_ai_message(
                    str(model_file),
                    messages,
                    tools_schema,
                    mode,
                    str(output_dir / "llm_calls"),
                    f"llm_call_{llm_call_start + llm_calls:03d}",
                )
                if not isinstance(llm_result, dict) or not isinstance(llm_result.get("ai_message"), dict):
                    raise ValueError("B4 result must contain an ai_message object")
                ai_message = llm_result["ai_message"]
                llm_status = llm_result.get("status")
                llm_error = llm_result.get("error")
            messages.append(ai_message)

        turn = {
            "turn_index": llm_call_start + llm_calls,
            "ai_message": ai_message,
            "llm_status": llm_status,
            "llm_error": llm_error,
            "tool_messages": [],
            "latency_ms": None,
        }
        if llm_status != "success":
            status = "llm_parse_error"
            terminal_error = {
                "type": "LLMParseError",
                "message": "B4 failed to parse the model output as a valid AIMessage JSON object.",
                "llm_call_index": llm_call_start + llm_calls,
                "cause": llm_error,
            }
            turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
            turns.append(turn)
            break

        tool_calls = ai_message.get("tool_calls", [])
        if not tool_calls:
            final_answer = ai_message["content"]
            print(f"content: {final_answer}")
            turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
            turns.append(turn)
            break

        # ── checkpoint: after_llm ──
        if ckpt_enabled and execution_mode == "integrated":
            _save_checkpoint(output_dir, {
                "checkpoint_version": 1,
                "conversation_id": runtime["conversation_id"],
                "stage": "after_llm",
                "messages": list(messages),
                "tool_rounds": tool_rounds,
                "llm_calls": llm_call_start + llm_calls,
                "pending_tool_calls": tool_calls,
                "output_dir": str(output_dir),
            })

        if tool_rounds >= runtime["max_turns"]:
            requested = ", ".join(call.get("name", "unknown") for call in tool_calls)
            final_answer = (
                "任务因超过最大工具调用轮次而终止，"
                f"最后一次模型仍请求调用工具：{requested}。"
            )
            status = "max_turns_exceeded"
            terminal_error = {
                "type": "MaxTurnsExceeded",
                "message": final_answer,
                "unexecuted_tool_calls": tool_calls,
            }
            turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
            turns.append(turn)
            break

        # ── Confirm mode: ask before executing dangerous tool calls ──
        if runtime.get("confirm_mode") and tool_calls:
            safe_tools = set(runtime.get("safe_tools", SAFE_TOOLS))
            dangerous = [c for c in tool_calls if c.get("name") not in safe_tools]
            if dangerous:
                plan_lines = [
                    f"  {j+1}. {c['name']}({__import__('json').dumps(c.get('args', {}), ensure_ascii=False)})"
                    for j, c in enumerate(tool_calls)
                ]
                flagged = ", ".join(c["name"] for c in dangerous)
                print(f"[Confirm] dangerous tool(s) detected: {flagged}")
                print(f"[Confirm] proposed calls:\n" + "\n".join(plan_lines))
                multi_cfg = runtime.get("multi_turn", {})
                if multi_cfg.get("interactive"):
                    confirm = input("[Confirm] execute? [Y/n]: ").strip().lower()
                    if confirm and confirm != "y":
                        final_answer = "用户取消了工具执行。"
                        status = "success"
                        turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
                        turns.append(turn)
                        break
            else:
                print(f"[Confirm] all tools safe, auto-approved")

        if execution_mode == "fixture":
            tool_messages = _fixture_tool_messages(
                tool_calls,
                fixture_data["tool_messages"],
            )
        else:
            for i, call in enumerate(tool_calls):
                tool_messages = execute_tool_calls(
                    [call],
                    str(tools_file),
                    runtime["toolset"],
                    str(output_dir),
                )
                # ── Innovation ⑤: auto-retry on recoverable errors ──
                if len(tool_messages) == 1:
                    err = _extract_tool_error(tool_messages[0])
                    if err:
                        fixed_args = _auto_fix_file_path(call.get("args", {}), err)
                        if fixed_args:
                            print(f"[AutoFix] {call['name']}: "
                                  f"{call.get('args', {}).get('file_path', '')} "
                                  f"→ {fixed_args.get('file_path', '')}")
                            tool_messages = execute_tool_calls(
                                [{**call, "args": fixed_args}],
                                str(tools_file),
                                runtime["toolset"],
                                str(output_dir),
                            )
                tool_rounds += 1
                messages.extend(tool_messages)
                all_tool_messages.extend(tool_messages)
                turn["tool_messages"].extend(tool_messages)
                # ── checkpoint: after_tool_{i} ──
                if ckpt_enabled:
                    _save_checkpoint(output_dir, {
                        "checkpoint_version": 1,
                        "conversation_id": runtime["conversation_id"],
                        "stage": f"after_tool_{i}",
                        "messages": list(messages),
                        "tool_rounds": tool_rounds,
                        "llm_calls": llm_call_start + llm_calls,
                        "pending_tool_calls": tool_calls[i + 1:] if i + 1 < len(tool_calls) else [],
                        "output_dir": str(output_dir),
                    })
            # In the fixture path or if tool_messages was set elsewhere
            if execution_mode == "fixture":
                tool_rounds += 1
                messages.extend(tool_messages)
                all_tool_messages.extend(tool_messages)
                turn["tool_messages"] = tool_messages
            turn["latency_ms"] = round((perf_counter() - turn_start) * 1000, 3)
            turns.append(turn)

    return {
        "status": status,
        "final_answer": final_answer,
        "tool_rounds": tool_rounds,
        "llm_calls": llm_calls,
        "turns": turns,
        "all_tool_messages": all_tool_messages,
        "terminal_error": terminal_error,
        "warnings": warnings,
    }


def _resolve_multi_turn_user_input(
    runtime: dict,
    round_idx: int,
) -> str | None:
    """Return the user input for round *round_idx* of a multi-turn conversation.

    Returns ``None`` when the conversation should end (EOF or exhausted queries).
    """
    multi_turn = runtime.get("multi_turn", {})
    if round_idx == 0:
        return runtime["user_input"]
    if multi_turn.get("interactive"):
        user_msg = input(f"[Round {round_idx + 1}] 请输入: ").strip()
        return user_msg if user_msg else None
    queries = multi_turn.get("follow_up_queries", [])
    if round_idx - 1 < len(queries):
        return queries[round_idx - 1]
    return None


def _check_prompt_switch(
    messages: list[dict],
    round_idx: int,
    switch_config: list[dict],
    prompt_dir: Path,
) -> tuple[str | None, str | None]:
    """Check whether a prompt template switch should fire for *round_idx*.

    Returns ``(new_prompt_text, action)`` when a rule triggers,
    or ``(None, None)`` otherwise.  The first matching rule wins.
    """
    for rule in switch_config:
        triggered = False
        if "after_round" in rule and round_idx >= rule["after_round"]:
            triggered = True
        if "on_keyword" in rule:
            keyword = rule["on_keyword"]
            for msg in reversed(messages):
                if msg["role"] == "user" and keyword in msg.get("content", ""):
                    triggered = True
                    break
        if triggered:
            template_path = prompt_dir / rule["template_path"]
            if template_path.exists():
                return read_text(template_path).strip(), rule.get("action", "append")
    return None, None


def _apply_prompt_switch(
    messages: list[dict],
    new_prompt: str,
    action: str,
) -> None:
    """Apply a prompt template switch, mutating *messages* in place.

    *action* must be ``"replace"`` (swap the first system message) or
    ``"append"`` (insert a new system message at the front).
    """
    system_indices = [i for i, m in enumerate(messages) if m["role"] == "system"]
    if action == "replace":
        if system_indices:
            messages[system_indices[0]]["content"] = new_prompt
    elif action == "append":
        messages.insert(0, {"role": "system", "content": new_prompt})


# ---------------------------------------------------------------------------
# Extension ④: history compression
# ---------------------------------------------------------------------------


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate for a message array.

    Chinese ≈1.5 chars/token, English ≈4 chars/token.
    We use //3 as a conservative midpoint so the threshold is reached
    *before* the actual context window fills up.
    """
    text = __import__("json").dumps(messages, ensure_ascii=False)
    return len(text) // 3


def _rule_based_summary(messages: list[dict]) -> str:
    """Mock-mode summary: extract user questions and assistant answers.

    Skips synthetic summary messages (``[历史对话摘要]``) to avoid
    recursive nesting across multiple compression rounds.
    """
    parts = []
    for msg in messages:
        content = msg.get("content", "")
        if msg["role"] == "user":
            if content.startswith("[历史对话摘要]"):
                parts.append(f"此前摘要: {content[:150]}")
            else:
                parts.append(f"用户提问: {content[:100]}")
        elif msg["role"] == "assistant" and content and not msg.get("tool_calls"):
            parts.append(f"回答要点: {content[:200]}")
    return "；".join(parts) if parts else "（无关键对话内容）"


def _summarize_messages(
    older_messages: list[dict],
    model_file: str,
    mode: str,
    output_dir: str,
) -> str:
    """Compress *older_messages* into a short summary string.

    In ``mock`` mode a rule-based extraction is used.  In ``prompt_json``
    mode B4 is asked to summarise the history.
    """
    if mode in ("mock", "fixture"):
        return _rule_based_summary(older_messages)

    from b4_local_agent_llm import generate_ai_message as b4_generate

    summary_prompt = (
        "请将以下对话历史压缩为一段简洁的摘要（中文，200字以内），"
        "保留关键任务、工具调用和结论：\n\n"
        + __import__("json").dumps(older_messages, ensure_ascii=False, indent=2)
    )
    result = b4_generate(
        model_file,
        [{"role": "user", "content": summary_prompt}],
        [],  # no tools needed for summarisation
        mode,
        output_dir,
        "compress_summary",
    )
    if result.get("status") == "success":
        return result["ai_message"]["content"]
    return "[对话历史摘要生成失败]"


def _compress_history_if_needed(
    messages: list[dict],
    config: dict,
    model_file: str,
    mode: str,
    output_dir: str,
) -> bool:
    """Compress older messages in-place when the token budget is exceeded.

    Strategy: keep all system messages + the most recent *keep_recent_rounds*
    worth of non-system messages; compress the middle portion into a summary.
    """
    compress_config = config.get("history_compression", {})
    if not compress_config.get("enabled"):
        return False

    max_tokens = compress_config.get("max_tokens", 3500)
    keep_recent = compress_config.get("keep_recent_rounds", 3)

    if _estimate_tokens(messages) <= max_tokens:
        return False

    # Locate the boundary between system messages and conversation body
    system_indices = [i for i, m in enumerate(messages) if m["role"] == "system"]
    last_system_idx = system_indices[-1] if system_indices else -1

    # Each round typically contributes ~3 messages (user, assistant, tool)
    recent_count = keep_recent * 3
    keep_from = max(last_system_idx + 1, len(messages) - recent_count)

    # Older messages: everything between the last system message and keep_from
    older = messages[last_system_idx + 1 : keep_from]
    older = [m for m in older if m["role"] != "system"]
    if not older:
        return False

    summary = _summarize_messages(older, model_file, mode, output_dir)

    # Rebuild: system messages + summary + recent messages
    rebuilt = messages[: last_system_idx + 1]
    rebuilt.append({
        "role": "user",
        "content": f"[历史对话摘要] {summary}",
    })
    rebuilt.extend(messages[keep_from:])

    old_count = len(messages) - len(rebuilt) + 1  # +1 for the summary message
    messages.clear()
    messages.extend(rebuilt)

    print(f"[Compress] → {len(messages)} messages "
          f"(condensed ~{old_count} older messages into summary)")
    return True


# ---------------------------------------------------------------------------
# Extension ②: checkpoint / resume
# ---------------------------------------------------------------------------

CHECKPOINT_FILE = "checkpoint.json"


def _save_checkpoint(outdir: Path, state: dict) -> None:
    """Persist a lightweight run-state snapshot."""
    state["timestamp"] = __import__("common.logging_utils", fromlist=["now_iso"]).now_iso()
    __import__("common.io_utils", fromlist=["write_json"]).write_json(
        state, outdir / CHECKPOINT_FILE,
    )


def _load_checkpoint(outdir: Path) -> dict | None:
    """Return the checkpoint dict if one exists, otherwise ``None``."""
    path = outdir / CHECKPOINT_FILE
    if path.exists():
        return __import__("common.io_utils", fromlist=["read_json"]).read_json(path)
    return None


# ---------------------------------------------------------------------------
# Innovation ⑤: tool result validation & auto-retry
# ---------------------------------------------------------------------------

def _auto_fix_file_path(args: dict, error_msg: str) -> dict | None:
    """Attempt common file-path corrections when a tool reports FileNotFoundError.

    Returns corrected *args* dict or ``None`` if the error is not automatically
    fixable.
    """
    if not isinstance(args, dict) or "file_path" not in args:
        return None
    if "FileNotFoundError" not in error_msg and "No such file" not in error_msg:
        return None

    file_path: str = args["file_path"]

    # Fix 1: strip accidental leading / trailing whitespace
    stripped = file_path.strip()
    if stripped != file_path:
        return {**args, "file_path": stripped}

    # Fix 2: try adding "docs/" prefix when the user omitted it
    if "/" not in file_path and not file_path.startswith("docs/"):
        return {**args, "file_path": f"docs/{file_path}"}

    return None


def _extract_tool_error(tool_message: dict) -> str | None:
    """Return the error string from a ToolMessage if its SkillResult has status=error."""
    try:
        content = tool_message.get("content", "")
        if isinstance(content, str):
            import json as _json
            result = _json.loads(content)
            if isinstance(result, dict) and result.get("status") == "error":
                return str(result.get("error", ""))
    except Exception:
        pass
    return None


def run_agent(
    input_path: str,
    tools_config: str | None,
    memory_config: str | None,
    model_config: str | None,
    outdir: str,
    llm_mode: str | None = None,
    resume: bool = False,
) -> dict:
    started = perf_counter()
    input_file = Path(input_path).resolve()
    output_dir = Path(outdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = _validate_runtime_input(read_json(input_file))
    print(f"user_input: {runtime['user_input']}")
    execution_mode = runtime["execution_mode"]
    prompt_path = resolve_from_file(runtime["system_prompt_path"], input_file)
    system_prompt = read_text(prompt_path).strip()
    fixture_data = None
    tools_file = memory_file = model_file = None
    execute_tool_calls_fn = None  # only assigned in integrated mode
    if execution_mode == "fixture":
        fixture_data = _load_fixture_inputs(input_file, runtime)
        selected_memory = fixture_data["selected_memory"]
        tools_schema = fixture_data["tools_schema"]
        mode = "fixture"
    else:
        if not tools_config or not memory_config or not model_config:
            raise ValueError("integrated mode requires tools_config, memory_config, and model_config")
        from b3_tool_layer import execute_tool_calls, get_tools_schema
        from b5_memory import load_memory

        execute_tool_calls_fn = execute_tool_calls
        tools_file = Path(tools_config).resolve()
        memory_file = Path(memory_config).resolve()
        model_file = Path(model_config).resolve()
        selected_memory = load_memory(
            str(memory_file),
            runtime["selected_memory_ids"],
            runtime["use_global_memory"],
            runtime["user_input"],
            str(output_dir),
        )
        tools_schema = get_tools_schema(str(tools_file), runtime["toolset"], str(output_dir))
        mode = llm_mode or _default_llm_mode(model_file)
    memory_context = _memory_context(selected_memory)
    if memory_context:
        system_prompt = f"{system_prompt}\n\n{memory_context}"

    # ---- determine conversation rounds ----
    multi_turn = runtime.get("multi_turn", {})
    max_rounds = multi_turn.get("max_rounds", 1) if multi_turn.get("enabled") else 1

    # ---- shared state across rounds ----
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    all_turns: list[dict] = []
    all_tool_messages: list[dict] = []
    round_summaries: list[dict] = []
    total_llm_calls = 0
    total_tool_rounds = 0
    final_answer = ""
    overall_status = "success"
    overall_terminal_error = None
    all_warnings: list[str] = []
    if selected_memory.get("status") in {"partial", "error"}:
        all_warnings.append("memory selection completed with errors")

    # ---- checkpoint resume (before multi-turn loop) ----
    resume_stage: str | None = None
    pending_tool_calls: list[dict] | None = None
    start_round = 0
    if resume:
        ckpt = _load_checkpoint(output_dir)
        if ckpt is None:
            raise ValueError(f"--resume requested but no {CHECKPOINT_FILE} found in {output_dir}")
        messages[:] = ckpt["messages"]
        total_llm_calls = ckpt.get("llm_calls", 0)
        total_tool_rounds = ckpt.get("tool_rounds", 0)
        resume_stage = ckpt["stage"]
        pending_tool_calls = ckpt.get("pending_tool_calls", [])
        print(f"[Checkpoint] resumed stage={resume_stage}, "
              f"llm_calls={total_llm_calls}, tool_rounds={total_tool_rounds}")
        # Determine which round we are on by counting user messages
        # Only count real user inputs, not plan-mode intermediate messages
        start_round = sum(
            1 for m in messages
            if m["role"] == "user"
            and "Plan" not in m.get("content", "")[:50]
        )
        # If we crashed mid-round (resume_stage is set), include the current
        # unfinished round so _agent_loop can pick up from the snapshot point
        if resume_stage:
            start_round = max(0, start_round - 1)
        # If resume_stage is "after_llm" or starts with "after_tool", we're mid-round
        if resume_stage == "after_llm" or resume_stage.startswith("after_tool"):
            pass  # _agent_loop will handle the resume

    for round_idx in range(start_round, max_rounds):
        user_msg = _resolve_multi_turn_user_input(runtime, round_idx)
        if user_msg is None:
            break  # EOF or exhausted follow_up_queries

        # ---- prompt template switch (before appending user message) ----
        prompt_config = runtime.get("prompt_templates", {})
        if prompt_config.get("switches"):
            new_prompt, action = _check_prompt_switch(
                messages, round_idx,
                prompt_config["switches"],
                input_file.parent,
            )
            if new_prompt:
                _apply_prompt_switch(messages, new_prompt, action)
                print(f"[PromptSwitch] round={round_idx}, action={action}")

        # On resume, the checkpoint already contains the user message for this round
        if not resume_stage:
            messages.append({"role": "user", "content": user_msg})
        print(f"[Round {round_idx + 1}/{max_rounds}] user_input: {user_msg[:80]}...")

        # ── Plan mode: two-phase (plan → approve → execute) ──
        # Skip plan phase on resume if plan was already approved before crash
        plan_already_approved = any(
            "Plan approved" in m.get("content", "") or "Plan auto-approved" in m.get("content", "")
            for m in messages if m["role"] == "user"
        )
        if runtime.get("plan_mode") and execution_mode != "fixture" and not plan_already_approved:
            plan_prompt = (
                "[PLAN MODE] Do NOT call any tools yet. "
                "First think through the request and describe your approach "
                "in natural language. What steps will you take? What tools "
                "might you need? Output as plain text, no JSON."
            )
            # Disable checkpoint during plan phase to avoid saving incomplete state
            plan_runtime = {**runtime, "checkpoint": {"enabled": False}}
            messages.append({"role": "user", "content": plan_prompt})
            plan_result = _agent_loop(
                messages=messages, tools_schema=[],  # no tools during plan phase
                runtime=plan_runtime, execution_mode=execution_mode,
                fixture_data=fixture_data, tools_file=tools_file,
                model_file=model_file, output_dir=output_dir,
                mode=mode, selected_memory=selected_memory,
                llm_call_start=total_llm_calls,
                execute_tool_calls=execute_tool_calls_fn,
            )
            total_llm_calls += plan_result["llm_calls"]
            # Remove the plan-mode instruction from messages
            messages.pop()  # pop the plan_prompt user message
            if plan_result.get("final_answer"):
                messages.pop()  # pop the assistant's plan response too

            plan_text = plan_result.get("final_answer", "（无法生成计划）")
            print(f"[Plan] LLM plan:\n{plan_text}")

            multi_cfg = runtime.get("multi_turn", {})
            plan_rejected = False
            if multi_cfg.get("interactive"):
                while True:
                    print(f"[Plan]\n{plan_text}")
                    action = input("[Plan] [A]pprove / [R]eject / [M]odify: ").strip().lower()
                    if action == "r":
                        final_answer = plan_text
                        overall_status = "success"
                        plan_rejected = True
                        round_summaries.append({
                            "round_idx": round_idx, "user_input": user_msg,
                            "status": "success", "final_answer": final_answer,
                            "llm_calls": plan_result["llm_calls"], "tool_rounds": 0,
                        })
                        break
                    elif action == "m":
                        feedback = input("[Plan] modification: ").strip()
                        messages.append({"role": "user",
                                         "content": (
                                             f"The plan was: {plan_text}\n\n"
                                             f"User feedback: {feedback}\n\n"
                                             "Please revise the plan accordingly. "
                                             "Output the revised plan as plain text. Do NOT call tools."
                                         )})
                        revise_result = _agent_loop(
                            messages=messages, tools_schema=[],
                            runtime=plan_runtime, execution_mode=execution_mode,
                            fixture_data=fixture_data, tools_file=tools_file,
                            model_file=model_file, output_dir=output_dir,
                            mode=mode, selected_memory=selected_memory,
                            llm_call_start=total_llm_calls,
                            execute_tool_calls=execute_tool_calls_fn,
                        )
                        total_llm_calls += revise_result["llm_calls"]
                        # Remove the revision request and assistant response
                        messages.pop()  # pop the feedback user message
                        if revise_result.get("final_answer"):
                            messages.pop()  # pop assistant's revised plan
                        plan_text = revise_result.get("final_answer", plan_text)
                        print(f"[Plan] LLM revised plan:\n{plan_text}")
                        # loop again to show revised plan and re-ask
                    else:  # 'a' or empty
                        messages.append({"role": "user",
                                         "content": f"Plan approved: {plan_text}\nNow execute using available tools."})
                        break
            else:
                # Non-interactive: auto-approve
                messages.append({"role": "user",
                                 "content": f"Plan auto-approved: {plan_text}\nNow execute using available tools."})
            if plan_rejected:
                break

            # Phase 2: execute (with tools enabled)
            loop_result = _agent_loop(
                messages=messages, tools_schema=tools_schema,
                runtime=runtime, execution_mode=execution_mode,
                fixture_data=fixture_data, tools_file=tools_file,
                model_file=model_file, output_dir=output_dir,
                mode=mode, selected_memory=selected_memory,
                llm_call_start=total_llm_calls,
                execute_tool_calls=execute_tool_calls_fn,
            )
        else:
            loop_result = _agent_loop(
                messages=messages,
                tools_schema=tools_schema,
                runtime=runtime,
                execution_mode=execution_mode,
                fixture_data=fixture_data,
                tools_file=tools_file,
                model_file=model_file,
                output_dir=output_dir,
                mode=mode,
                selected_memory=selected_memory,
                llm_call_start=total_llm_calls,
                execute_tool_calls=execute_tool_calls_fn,
                resume_stage=resume_stage,
                pending_tool_calls=pending_tool_calls,
            )
        # Consume resume state so subsequent rounds start fresh
        resume_stage = None
        pending_tool_calls = None

        all_turns.extend(loop_result["turns"])
        all_tool_messages.extend(loop_result["all_tool_messages"])
        total_llm_calls += loop_result["llm_calls"]
        total_tool_rounds += loop_result["tool_rounds"]
        all_warnings.extend(loop_result["warnings"])
        round_summaries.append({
            "round_idx": round_idx,
            "user_input": user_msg,
            "status": loop_result["status"],
            "final_answer": loop_result["final_answer"],
            "llm_calls": loop_result["llm_calls"],
            "tool_rounds": loop_result["tool_rounds"],
        })

        if loop_result["status"] != "success":
            final_answer = loop_result["final_answer"]
            overall_status = loop_result["status"]
            overall_terminal_error = loop_result["terminal_error"]
            break

        final_answer = loop_result["final_answer"]

        # ---- history compression (after successful round) ----
        _compress_history_if_needed(
            messages, runtime,
            str(model_file) if model_file else "",
            mode,
            str(output_dir),
        )

    # ---- output artifacts (compatible with single-round format) ----
    write_json(messages, output_dir / "messages.json")
    if execution_mode == "integrated":
        write_json(all_tool_messages, output_dir / "tool_messages.json")
    write_text(final_answer.strip() + "\n", output_dir / "final_answer.md")

    # Build memory_save decision
    memory_save = {"requested": runtime["save_memory"], "status": "not_requested"}
    if overall_status != "success" and runtime["save_memory"] != "none":
        memory_save = {"requested": runtime["save_memory"], "status": "skipped", "reason": overall_status}

    # Build trace (add rounds info when multi-turn is active)
    trace = {
        "conversation_id": runtime["conversation_id"],
        "execution_mode": execution_mode,
        "status": overall_status,
        "toolset": runtime["toolset"],
        "max_turns": runtime["max_turns"],
        "tool_rounds_used": total_tool_rounds,
        "llm_call_count": total_llm_calls,
        "turns": all_turns,
        "final_answer_path": "final_answer.md",
        "memory_save": memory_save,
        "warnings": all_warnings,
        "error": overall_terminal_error,
    }
    if multi_turn.get("enabled"):
        trace["multi_turn"] = {
            "enabled": True,
            "max_rounds": max_rounds,
            "completed_rounds": len(round_summaries),
            "rounds": round_summaries,
        }
    write_json(trace, output_dir / "trace.json")

    saved_memory = None
    if execution_mode == "integrated" and runtime["save_memory"] != "none" and trace["status"] == "success":
        try:
            from b5_memory import save_memory

            saved_memory = save_memory(
                str(memory_file),
                runtime["conversation_id"],
                runtime["save_memory"],
                str(output_dir / "messages.json"),
                str(output_dir / "trace.json"),
                str(output_dir / "final_answer.md"),
                str(output_dir),
            )
            trace["memory_save"] = {"requested": runtime["save_memory"], "status": "success"}
        except Exception as exc:
            trace["memory_save"] = {
                "requested": runtime["save_memory"],
                "status": "error",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            trace["warnings"].append("memory save failed")
            if trace["status"] == "success":
                trace["status"] = "partial"
        write_json(trace, output_dir / "trace.json")

    # ---- checkpoint cleanup on success ----
    if trace["status"] == "success":
        checkpoint_path = output_dir / CHECKPOINT_FILE
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    result = {
        "conversation_id": runtime["conversation_id"],
        "execution_mode": execution_mode,
        "status": trace["status"],
        "final_answer": final_answer,
        "messages_path": str(output_dir / "messages.json"),
        "trace_path": str(output_dir / "trace.json"),
        "final_answer_path": str(output_dir / "final_answer.md"),
        "selected_memory": selected_memory,
        "saved_memory": saved_memory,
        "elapsed_ms": round((perf_counter() - started) * 1000, 3),
    }
    if execution_mode == "integrated":
        append_jsonl(
            {
                "timestamp": now_iso(),
                "conversation_id": runtime["conversation_id"],
                "execution_mode": execution_mode,
                "status": trace["status"],
                "llm_mode": mode,
                "tool_rounds_used": total_tool_rounds,
                "llm_call_count": total_llm_calls,
                "elapsed_ms": result["elapsed_ms"],
            },
            output_dir / "runtime_log.jsonl",
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Agent message and tool loop.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--tools_config")
    parser.add_argument("--memory_config")
    parser.add_argument("--model_config")
    parser.add_argument("--llm_mode", choices=["mock", "prompt_json"], default=None)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--resume", action="store_true", default=False,
                        help="resume from checkpoint.json if present")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_agent(
            str(resolve_cli_path(args.input)),
            str(resolve_cli_path(args.tools_config)) if args.tools_config else None,
            str(resolve_cli_path(args.memory_config)) if args.memory_config else None,
            str(resolve_cli_path(args.model_config)) if args.model_config else None,
            str(resolve_cli_path(args.outdir)),
            args.llm_mode,
            resume=args.resume,
        )
        print(result["final_answer_path"])
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
