from __future__ import annotations

import csv
import statistics

from skills import resolve_data_path
from skills.error_codes import (
    ERR_FILE_NOT_FOUND,
    ERR_FILE_TYPE,
    ERR_FORMAT_INVALID,
    ERR_PARAM_RANGE,
    ERR_PARAM_TYPE,
    SkillError,
)


def table_analyzer(
    path: str,
    max_rows_preview: int = 5,
    describe: bool = True,
    *,
    data_root: str | None = None,
) -> dict:
    if not isinstance(max_rows_preview, int) or isinstance(max_rows_preview, bool):
        raise SkillError(ERR_PARAM_TYPE, f"max_rows_preview must be an integer, got {type(max_rows_preview).__name__}", {"param": "max_rows_preview", "value": max_rows_preview})
    if max_rows_preview < 0:
        raise SkillError(ERR_PARAM_RANGE, "max_rows_preview must be a non-negative integer", {"param": "max_rows_preview", "value": max_rows_preview})
    source, root = resolve_data_path(path, data_root)
    if source.suffix.lower() not in {".csv", ".tsv"}:
        raise SkillError(ERR_FILE_TYPE, "table_analyzer only supports .csv and .tsv files", {"path": path})
    if not source.is_file():
        raise SkillError(ERR_FILE_NOT_FOUND, f"table file not found: {path}", {"path": path})
    delimiter = "\t" if source.suffix.lower() == ".tsv" else ","
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise SkillError(ERR_FORMAT_INVALID, "table must contain a header row", {"path": path})
        rows = list(reader)
        columns = list(reader.fieldnames)
    stats: dict[str, dict] = {}
    if describe:
        for column in columns:
            raw_values = [row.get(column, "").strip() for row in rows]
            if not raw_values or any(value == "" for value in raw_values):
                continue
            try:
                values = [float(value) for value in raw_values]
            except ValueError:
                continue
            stats[column] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.fmean(values),
            }
    return {
        "path": source.relative_to(root).as_posix(),
        "num_rows": len(rows),
        "num_columns": len(columns),
        "columns": columns,
        "preview": rows[:max_rows_preview],
        "describe": stats,
    }