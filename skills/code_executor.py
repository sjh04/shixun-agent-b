from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from skills.error_codes import (
        ERR_CALCULATION,
        ERR_PARAM_MISSING,
        ERR_PARAM_RANGE,
        ERR_PARAM_TYPE,
        ERR_SANDBOX,
        ERR_TIMEOUT,
        ERR_INTERNAL,
        SkillError,
    )
except ImportError:
    ERR_PARAM_MISSING, ERR_PARAM_RANGE, ERR_PARAM_TYPE = 1001, 1003, 1002
    ERR_CALCULATION = 3001
    ERR_TIMEOUT = 3003
    ERR_SANDBOX = 3004
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


ALLOWED_NODES = {
    ast.Module,
    ast.Import, ast.ImportFrom, ast.alias,
    ast.Expr, ast.Assign, ast.AnnAssign, ast.AugAssign,
    ast.Return, ast.Pass, ast.Break, ast.Continue,
    ast.If, ast.For, ast.While,
    ast.Try,ast.ExceptHandler, ast.Raise, ast.Assert,
    ast.With, ast.withitem,
    ast.FunctionDef, ast.arguments, ast.arg,
    ast.BoolOp, ast.IfExp,
    ast.NamedExpr,
    ast.Lambda,
    ast.Call, ast.keyword,
    ast.Attribute, ast.Subscript, ast.Slice,
    ast.Name, ast.Constant, ast.JoinedStr, ast.FormattedValue,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    ast.comprehension,
    ast.List, ast.Tuple, ast.Set, ast.Dict,
    ast.Compare,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
    ast.FloorDiv, ast.MatMult,
    ast.And, ast.Or,
    ast.Not, ast.Invert, ast.UAdd, ast.USub,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn,
    ast.UnaryOp, ast.BinOp,
    ast.Load, ast.Store, ast.Del,
    ast.Starred,
}

ALLOWED_IMPORTS = {
    "math", "statistics", "json", "csv", "re",
    "datetime", "collections", "itertools", "functools",
    "decimal", "fractions", "random", "string", "textwrap",
    "hashlib", "base64", "uuid", "dataclasses",
}

FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "__import__",
    "open", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "hasattr",
    "type", "isinstance", "issubclass",
    "super", "object", "breakpoint", "input",
    "memoryview", "__builtins__",
}

FORBIDDEN_PREFIXES = (
    "os", "sys", "subprocess", "shutil", "pathlib",
    "importlib", "inspect", "ctypes", "socket",
    "http", "urllib", "ftplib", "telnetlib",
    "pickle", "marshal", "code",
)


def _ast_check(code: str) -> str | None:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"syntax error: {exc}"

    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            return f"forbidden node type: {type(node).__name__}"

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            else:
                modules = [node.module.split(".")[0]] if node.module else []
            for mod in modules:
                if mod not in ALLOWED_IMPORTS:
                    return f"import blocked: '{mod}' is not in allowed modules"

        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_NAMES:
                return f"forbidden function call: {func.id}"
            if isinstance(func, ast.Attribute):
                value = func.value
                if isinstance(value, ast.Name) and value.id in FORBIDDEN_PREFIXES:
                    return f"forbidden module access: {value.id}.{func.attr}"

        if isinstance(node, ast.Attribute):
            value = node.value
            if isinstance(value, ast.Name) and value.id in FORBIDDEN_PREFIXES:
                return f"forbidden attribute access: {value.id}.{node.attr}"

    return None


MAX_OUTPUT_CHARS = 10000
DEFAULT_TIMEOUT = 5


def _run_in_subprocess(code: str, timeout_sec: float) -> dict:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", encoding="utf-8", delete=False
        ) as tmp:
            tmp.write(code)
            tmp_path = Path(tmp.name)

        result = subprocess.run(
            [sys.executable, str(tmp_path)],
            timeout=timeout_sec,
            capture_output=True,
            text=True,
        )

        raw_stdout = result.stdout or ""
        raw_stderr = result.stderr or ""

        return {
            "returncode": result.returncode,
            "stdout": raw_stdout[:MAX_OUTPUT_CHARS],
            "stderr": raw_stderr[:MAX_OUTPUT_CHARS],
            "stdout_truncated": len(raw_stdout) > MAX_OUTPUT_CHARS,
            "stderr_truncated": len(raw_stderr) > MAX_OUTPUT_CHARS,
        }

    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"execution timed out after {timeout_sec}s",
            "stdout_truncated": False,
            "stderr_truncated": False,
        }
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def code_executor(
    code: str,
    timeout_sec: float = DEFAULT_TIMEOUT,
) -> dict:
    if not isinstance(code, str) or not code.strip():
        raise SkillError(
            ERR_PARAM_MISSING,
            "code must be a non-empty string",
            {"param": "code"},
        )
    if not isinstance(timeout_sec, (int, float)) or isinstance(timeout_sec, bool):
        raise SkillError(
            ERR_PARAM_TYPE,
            f"timeout_sec must be a number, got {type(timeout_sec).__name__}",
            {"param": "timeout_sec", "value": timeout_sec},
        )
    if timeout_sec <= 0 or timeout_sec > 30:
        raise SkillError(
            ERR_PARAM_RANGE,
            "timeout_sec must be > 0 and ≤ 30",
            {"param": "timeout_sec", "value": timeout_sec},
        )
    if len(code) > 50000:
        raise SkillError(
            ERR_PARAM_RANGE,
            "code must not exceed 50000 characters",
            {"param": "code", "length": len(code)},
        )

    violation = _ast_check(code)
    if violation is not None:
        raise SkillError(
            ERR_SANDBOX,
            f"sandbox violation: {violation}",
            {"violation": violation},
        )

    exec_result = _run_in_subprocess(code, timeout_sec)

    if exec_result["returncode"] == -1:
        raise SkillError(
            ERR_TIMEOUT,
            f"code execution timed out after {timeout_sec}s",
            {"timeout_sec": timeout_sec},
        )

    if exec_result["returncode"] != 0:
        raise SkillError(
            ERR_CALCULATION,
            f"code exited with code {exec_result['returncode']}",
            {
                "returncode": exec_result["returncode"],
                "stderr": exec_result["stderr"].strip(),
            },
        )

    return {
        "stdout": exec_result["stdout"],
        "stderr": exec_result["stderr"],
        "returncode": exec_result["returncode"],
        "stdout_truncated": exec_result["stdout_truncated"],
        "stderr_truncated": exec_result["stderr_truncated"],
        "execution_method": "subprocess",
    }