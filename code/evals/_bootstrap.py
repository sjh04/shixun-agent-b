from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_code_path() -> Path:
    """Add agent/code to sys.path for eval scripts stored under agent/code/evals."""
    code_root = Path(__file__).resolve().parents[1]
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    return code_root
