from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from common.io_utils import append_jsonl, read_json, read_yaml, write_json
from common.logging_utils import now_iso
from common.path_utils import bootstrap_project_root, resolve_cli_path, resolve_from_file
from common.schemas import make_skill_result, make_tool_message, normalize_tool_call


bootstrap_project_root()


JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}

# 进阶 2: 可恢复错误 = 瞬时/外部性错误, 允许有限重试; 参数错误(ValueError)等不可恢复错误不重试。
# 通过异常类名匹配, 避免 B3 与具体 Skill 强耦合。
RECOVERABLE_EXCEPTIONS = (TimeoutError, ConnectionError, InterruptedError)
RECOVERABLE_EXCEPTION_NAMES = {"RecoverableToolError"}


def _is_recoverable(exc: Exception) -> bool:
    if isinstance(exc, RECOVERABLE_EXCEPTIONS):
        return True
    return type(exc).__name__ in RECOVERABLE_EXCEPTION_NAMES


def _cache_key(name: str, args: dict) -> str:
    return name + "|" + json.dumps(args, sort_keys=True, ensure_ascii=False)


def _load_tools_config(tools_config: str | Path) -> tuple[Path, dict]:
    config_path = Path(tools_config).resolve()
    config = read_yaml(config_path)
    if not isinstance(config, dict):
        raise ValueError("tools.yaml must contain an object")
    if not isinstance(config.get("tools"), dict) or not isinstance(config.get("toolsets"), dict):
        raise ValueError("tools.yaml must define tools and toolsets")
    return config_path, config


def _resolve_toolset(config: dict, toolset: str | None) -> tuple[str, list[str]]:
    selected = toolset or config.get("default_toolset")
    if not isinstance(selected, str) or selected not in config["toolsets"]:
        raise ValueError(f"toolset does not exist: {selected}")
    names = config["toolsets"][selected]
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"toolset {selected} must be a list of tool names")
    return selected, names


def _parameter_schema(tool: dict) -> dict:
    raw_parameters = tool.get("parameters", {})
    if not isinstance(raw_parameters, dict):
        raise ValueError("tool parameters must be an object")
    properties = {}
    for name, definition in raw_parameters.items():
        if not isinstance(definition, dict) or definition.get("type") not in JSON_TYPES:
            raise ValueError(f"invalid parameter schema for {name}")
        properties[name] = dict(definition)
    required = tool.get("required", [])
    if not isinstance(required, list) or not all(name in properties for name in required):
        raise ValueError("required parameters must reference declared properties")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def get_tools_schema(
    tools_config: str,
    toolset: str,
    outdir: str | None = None,
) -> list[dict]:
    _, config = _load_tools_config(tools_config)
    selected, tool_names = _resolve_toolset(config, toolset)
    schema = []
    for name in tool_names:
        tool = config["tools"].get(name)
        if not isinstance(tool, dict):
            raise ValueError(f"toolset references missing tool: {name}")
        for field in ("module", "function", "description", "returns"):
            if field not in tool:
                raise ValueError(f"tool {name} missing {field}")
        returns = tool["returns"]
        if not isinstance(returns, dict):
            raise ValueError(f"tool {name} returns must be an object")
        schema.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool["description"],
                    "parameters": _parameter_schema(tool),
                    "x-returns": {"type": "object", "properties": returns},
                },
            }
        )
    if outdir:
        output_dir = Path(outdir)
        write_json(schema, output_dir / "tools_schema.json")
        write_json(
            {"status": "success", "toolset": selected, "tool_count": len(schema), "tools": tool_names},
            output_dir / "tool_schema_report.json",
        )
    return schema


def _validate_args(args: dict, definition: dict) -> None:
    parameter_schema = _parameter_schema(definition)
    properties = parameter_schema["properties"]
    missing = [name for name in parameter_schema["required"] if name not in args]
    if missing:
        raise ValueError(f"missing required parameters: {', '.join(missing)}")
    unknown = sorted(set(args) - set(properties))
    if unknown:
        raise ValueError(f"unknown parameters: {', '.join(unknown)}")
    for name, value in args.items():
        expected_name = properties[name]["type"]
        expected = JSON_TYPES[expected_name]
        if expected_name in {"integer", "number"} and isinstance(value, bool):
            valid = False
        else:
            valid = isinstance(value, expected)
        if not valid:
            raise ValueError(f"parameter {name} must be {expected_name}")
        if expected_name == "array" and "items" in properties[name]:
            item_type = properties[name]["items"].get("type")
            if item_type in JSON_TYPES and not all(isinstance(item, JSON_TYPES[item_type]) for item in value):
                raise ValueError(f"parameter {name} contains invalid items")


def _error_result(name: str, args: dict, exc: Exception, latency_ms: float = 0.0) -> dict:
    return make_skill_result(
        name,
        "error",
        args,
        None,
        {"type": type(exc).__name__, "message": str(exc)},
        latency_ms,
    )


def _run_with_retry(
    function: Any,
    kwargs: dict,
    name: str,
    args: dict,
    max_retries: int,
) -> tuple[dict, int]:
    """执行 Skill, 对可恢复错误做有限重试(进阶 2)。返回 (SkillResult, attempts)。"""
    attempt = 0
    last_exc: Exception | None = None
    while attempt <= max_retries:
        attempt += 1
        start = perf_counter()
        try:
            output = function(**kwargs)
            latency_ms = round((perf_counter() - start) * 1000, 3)
            return make_skill_result(name, "success", args, output, None, latency_ms), attempt
        except (ImportError, AttributeError):
            raise
        except Exception as exc:  # noqa: BLE001 - 分类由 _is_recoverable 决定
            last_exc = exc
            latency_ms = round((perf_counter() - start) * 1000, 3)
            if _is_recoverable(exc) and attempt <= max_retries:
                continue
            return _error_result(name, args, exc, latency_ms), attempt
    # 理论上不可达; 兜底返回最后一次错误。
    return _error_result(name, args, last_exc or RuntimeError("unknown error")), attempt


def _aggregate_stats(log_records: list[dict], toolset: str) -> dict:
    """按工具名聚合调用次数/失败率/平均耗时(进阶 4)。"""
    per_tool: dict[str, dict] = {}
    for record in log_records:
        name = record["name"]
        bucket = per_tool.setdefault(
            name,
            {"count": 0, "success": 0, "error": 0, "cache_hits": 0, "_latencies": []},
        )
        bucket["count"] += 1
        bucket["success" if record["status"] == "success" else "error"] += 1
        if record.get("cache_hit"):
            bucket["cache_hits"] += 1
        latency = record.get("latency_ms")
        if isinstance(latency, (int, float)):
            bucket["_latencies"].append(latency)

    tools_stats = {}
    total_calls = total_success = total_error = 0
    for name, bucket in per_tool.items():
        latencies = bucket.pop("_latencies")
        count = bucket["count"]
        tools_stats[name] = {
            "count": count,
            "success": bucket["success"],
            "error": bucket["error"],
            "cache_hits": bucket["cache_hits"],
            "failure_rate": round(bucket["error"] / count, 4) if count else 0.0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "max_latency_ms": round(max(latencies), 3) if latencies else 0.0,
        }
        total_calls += count
        total_success += bucket["success"]
        total_error += bucket["error"]

    return {
        "toolset": toolset,
        "total_calls": total_calls,
        "total_success": total_success,
        "total_error": total_error,
        "overall_failure_rate": round(total_error / total_calls, 4) if total_calls else 0.0,
        "tools": tools_stats,
    }


def execute_tool_calls(
    tool_calls: list[dict],
    tools_config: str,
    toolset: str | None = None,
    outdir: str | None = None,
) -> list[dict]:
    config_path, config = _load_tools_config(tools_config)
    selected, allowed_tools = _resolve_toolset(config, toolset)
    if not isinstance(tool_calls, list):
        raise ValueError("tool_calls must be a list")
    settings = config.get("settings", {})
    data_root_setting = settings.get("data_root", "../data")
    resolved_data_root = resolve_from_file(data_root_setting, config_path)
    max_retries = settings.get("max_retries", 0)
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
        raise ValueError("settings.max_retries must be a non-negative integer")
    cache_enabled = bool(settings.get("cache_enabled", False))
    result_cache: dict[str, dict] = {}  # 进阶 3: 相同 name+args 复用成功结果。
    tool_messages = []
    log_records = []
    output_dir = Path(outdir) if outdir else None
    for index, raw_call in enumerate(tool_calls):
        attempts = 1
        cache_hit = False
        start = perf_counter()
        try:
            call = normalize_tool_call(raw_call, index)
        except Exception as exc:
            call = {"id": f"call_{index + 1:03d}", "name": "unknown", "args": {}}
            result = _error_result(call["name"], call["args"], exc)
        else:
            name = call["name"]
            args = call["args"]
            if name not in allowed_tools or name not in config["tools"]:
                result = _error_result(name, args, ValueError(f"tool is not available in {selected}: {name}"))
            else:
                definition = config["tools"][name]
                cacheable = cache_enabled and definition.get("cacheable", True)
                key = _cache_key(name, args) if cacheable else None
                if key is not None and key in result_cache:
                    result = dict(result_cache[key])
                    cache_hit = True
                else:
                    try:
                        _validate_args(args, definition)
                        module = importlib.import_module(definition["module"])
                        function = getattr(module, definition["function"])
                        kwargs = dict(args)
                        signature = inspect.signature(function)
                        if "data_root" in signature.parameters:
                            kwargs["data_root"] = str(resolved_data_root)
                        if "output_dir" in signature.parameters:
                            kwargs["output_dir"] = str(output_dir) if output_dir else None
                        # 仅对声明 retryable 的工具启用重试; 其余工具尝试一次。
                        tool_retries = max_retries if definition.get("retryable") else 0
                        result, attempts = _run_with_retry(function, kwargs, name, args, tool_retries)
                    except (ImportError, AttributeError) as exc:
                        raise RuntimeError(f"cannot load configured tool {name}: {exc}") from exc
                    except Exception as exc:
                        latency_ms = round((perf_counter() - start) * 1000, 3)
                        result = _error_result(name, args, exc, latency_ms)
                    if key is not None and result["status"] == "success":
                        result_cache[key] = dict(result)
        content = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        message = make_tool_message(call["id"], call["name"], content, result["status"])
        tool_messages.append(message)
        log_records.append(
            {
                "timestamp": now_iso(),
                "toolset": selected,
                "tool_call_id": call["id"],
                "name": call["name"],
                "status": result["status"],
                "args": call["args"],
                "attempts": attempts,
                "cache_hit": cache_hit,
                "skill_result": result,
                "latency_ms": result["latency_ms"],
            }
        )
    if outdir:
        write_json(tool_messages, output_dir / "tool_messages.json")
        write_json(_aggregate_stats(log_records, selected), output_dir / "tool_call_stats.json")
        for record in log_records:
            append_jsonl(record, output_dir / "tool_call_log.jsonl")
    return tool_messages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate tool schema or execute tool calls.")
    parser.add_argument("--tools_config", required=True)
    parser.add_argument("--toolset", default=None)
    parser.add_argument("--tool_calls")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--export_schema", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--outdir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = resolve_cli_path(args.tools_config)
        outdir = resolve_cli_path(args.outdir)
        if args.export_schema:
            if not args.toolset:
                _, config = _load_tools_config(config_path)
                args.toolset = config.get("default_toolset")
            get_tools_schema(str(config_path), args.toolset, str(outdir))
            print(outdir / "tools_schema.json")
        else:
            if not args.tool_calls:
                raise ValueError("--tool_calls is required with --execute")
            payload = read_json(resolve_cli_path(args.tool_calls))
            tool_calls = payload.get("tool_calls") if isinstance(payload, dict) else payload
            execute_tool_calls(tool_calls, str(config_path), args.toolset, str(outdir))
            print(outdir / "tool_messages.json")
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
