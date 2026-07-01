"""B2 Skill 工具函数包。

提供 Skill 函数共用的基础设施：
  - DEFAULT_DATA_ROOT : 默认数据根目录（B2/data/）
  - resolve_data_path : 安全路径解析（防越权）
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def resolve_data_path(path: str, data_root: str | None = None) -> tuple[Path, Path]:
    root = Path(data_root).resolve() if data_root else DEFAULT_DATA_ROOT.resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes data root: {path}") from exc
    return candidate, root