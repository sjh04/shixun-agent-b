from __future__ import annotations

import importlib
import inspect
from pathlib import Path

try:
    from skills.error_codes import (
        ERR_PIPELINE_UNKNOWN_SKILL,
        ERR_PIPELINE_MISMATCH,
        ERR_UNKNOWN_SKILL,
        ERR_IMPORT,
        ERR_INTERNAL,
        SkillError,
    )
except ImportError:
    ERR_PIPELINE_UNKNOWN_SKILL = 5001
    ERR_PIPELINE_MISMATCH = 5002
    ERR_UNKNOWN_SKILL = 9001
    ERR_IMPORT = 9002
    ERR_INTERNAL = 9999

    class SkillError(Exception):
        def __init__(self, code: int, message: str, detail: dict | None = None) -> None:
            self.code = code
            self.message = message
            self.detail = detail or {}
            super().__init__(message)

        def as_error_dict(self) -> dict:
            err: dict = {
                "code": self.code,
                "message": self.message,
            }
            if self.detail:
                err["detail"] = self.detail
            return err


_SKILL_MODULES = {
    "calculator": "skills.calculator",
    "code_executor": "skills.code_executor",
    "local_file_search": "skills.local_file_search",
    "file_reader": "skills.file_reader",
    "format_converter": "skills.format_converter",
    "table_analyzer": "skills.table_analyzer",
}

INPUT_MAP: dict[tuple[str, str], dict[str, str]] = {
    ("local_file_search", "file_reader"): {"results[0].path": "path"},
    ("file_reader", "format_converter"): {"content": "text"},
    ("local_file_search", "table_analyzer"): {"results[0].path": "path"},
}

PIPELINES: dict[str, list[str]] = {
    "search_and_read": ["local_file_search", "file_reader"],
    "read_and_convert": ["file_reader", "format_converter"],
    "search_and_analyze": ["local_file_search", "table_analyzer"],
    "search_read_convert": ["local_file_search", "file_reader", "format_converter"],
}


def _get_nested(data: dict, path: str):
    normalized = path.replace("[", ".").replace("]", "")
    parts = normalized.split(".")
    current = data
    for part in parts:
        if isinstance(current, list) and part.isdigit():
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current[part]
        else:
            raise KeyError(
                f"Cannot access '{part}' in {type(current).__name__}: path='{path}'"
            )
    return current


def _load_skill_function(skill_name: str):
    if skill_name not in _SKILL_MODULES:
        raise SkillError(
            ERR_UNKNOWN_SKILL,
            f"unknown skill in pipeline: {skill_name}",
            {"skill_name": skill_name, "registered": list(_SKILL_MODULES.keys())},
        )
    try:
        module = importlib.import_module(_SKILL_MODULES[skill_name])
    except ImportError as exc:
        raise SkillError(
            ERR_IMPORT,
            f"failed to import {skill_name}: {exc}",
            {"skill_name": skill_name, "module": _SKILL_MODULES[skill_name]},
        ) from exc
    return getattr(module, skill_name)


def _map_fields(
    upstream_output: dict,
    upstream_skill: str,
    downstream_skill: str,
) -> dict:
    mapping = INPUT_MAP.get((upstream_skill, downstream_skill))
    if mapping is None:
        raise SkillError(
            ERR_PIPELINE_MISMATCH,
            f"no field mapping defined for {upstream_skill} → {downstream_skill}",
            {
                "upstream_skill": upstream_skill,
                "downstream_skill": downstream_skill,
                "available_pairs": [list(pair) for pair in INPUT_MAP.keys()],
            },
        )
    downstream_input: dict = {}
    for upstream_path, downstream_key in mapping.items():
        try:
            downstream_input[downstream_key] = _get_nested(upstream_output, upstream_path)
        except (KeyError, IndexError, TypeError) as exc:
            raise SkillError(
                ERR_PIPELINE_MISMATCH,
                f"field mapping failed: {upstream_skill}.{upstream_path} "
                f"→ {downstream_skill}.{downstream_key}: {exc}",
                {
                    "upstream_skill": upstream_skill,
                    "upstream_path": upstream_path,
                    "downstream_skill": downstream_skill,
                    "downstream_key": downstream_key,
                },
            ) from exc
    return downstream_input


def run_composite_skill(
    pipeline_name: str,
    input_data: dict,
    data_root: str | None = None,
    output_dir: str | None = None,
) -> dict:
    if not isinstance(input_data, dict):
        raise ValueError("pipeline input must be a JSON object (dict)")

    if pipeline_name not in PIPELINES:
        raise SkillError(
            ERR_PIPELINE_UNKNOWN_SKILL,
            f"unknown pipeline: {pipeline_name}",
            {
                "pipeline_name": pipeline_name,
                "available": sorted(PIPELINES.keys()),
            },
        )

    steps = PIPELINES[pipeline_name]
    trace: list[dict] = []
    current_input = dict(input_data)

    for i, skill_name in enumerate(steps):
        func = _load_skill_function(skill_name)

        signature = inspect.signature(func)
        sig_params = set(signature.parameters.keys())

        kwargs: dict = {}
        for key, value in current_input.items():
            if key in sig_params:
                kwargs[key] = value
        for key, value in input_data.items():
            if key not in kwargs and key in sig_params:
                kwargs[key] = value

        if "data_root" in sig_params:
            kwargs["data_root"] = data_root
        if "output_dir" in sig_params:
            kwargs["output_dir"] = output_dir

        try:
            output = func(**kwargs)
            trace.append({
                "step": i + 1,
                "skill": skill_name,
                "status": "success",
                "output": output,
                "error": None,
            })
        except Exception as exc:
            if isinstance(exc, SkillError):
                error_info = exc.as_error_dict()
            else:
                error_info = {"type": type(exc).__name__, "message": str(exc)}

            trace.append({
                "step": i + 1,
                "skill": skill_name,
                "status": "error",
                "output": None,
                "error": error_info,
            })
            return {
                "pipeline": pipeline_name,
                "status": "error",
                "failed_at": skill_name,
                "final_output": None,
                "trace": trace,
            }

        if i < len(steps) - 1:
            next_skill = steps[i + 1]
            current_input = _map_fields(output, skill_name, next_skill)

    return {
        "pipeline": pipeline_name,
        "status": "success",
        "final_output": trace[-1]["output"],
        "trace": trace,
    }