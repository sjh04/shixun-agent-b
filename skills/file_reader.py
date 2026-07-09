from __future__ import annotations

from skills import resolve_data_path
from skills.error_codes import (
    ERR_FILE_NOT_FOUND,
    ERR_FILE_TYPE,
    ERR_PARAM_RANGE,
    ERR_PARAM_TYPE,
    SkillError,
)


def file_reader(path: str, max_chars: int = 2000, *, data_root: str | None = None) -> dict:
    if not isinstance(max_chars, int) or isinstance(max_chars, bool):
        raise SkillError(ERR_PARAM_TYPE, f"max_chars must be an integer, got {type(max_chars).__name__}", {"param": "max_chars", "value": max_chars})
    if max_chars <= 0:
        raise SkillError(ERR_PARAM_RANGE, "max_chars must be a positive integer", {"param": "max_chars", "value": max_chars})
    source, root = resolve_data_path(path, data_root)
    if source.suffix.lower() not in {".txt", ".md"}:
        raise SkillError(ERR_FILE_TYPE, "file_reader only supports .txt and .md files", {"path": path})
    if not source.is_file():
        raise SkillError(ERR_FILE_NOT_FOUND, f"file not found: {path}", {"path": path})
    original = source.read_text(encoding="utf-8")
    content = original[:max_chars]
    return {
        "content": content,
        "num_chars": len(content),
        "source": source.relative_to(root).as_posix(),
        "truncated": len(original) > len(content),
    }