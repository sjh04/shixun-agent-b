"""B2 Skill：安全算术表达式求值器。

基于 AST 白名单的安全计算器，仅支持：
  - 四则运算：+ - * / // % **
  - 一元正负号：+x -x
  - 数字字面量：整数、浮点数
  - 常用数学函数：abs/min/max/round（通过 math 模块）

不支持：变量、函数调用、导入、位运算等。
"""

from __future__ import annotations

import ast
import math
import operator

from skills.error_codes import (
    ERR_CALCULATION,
    ERR_PARAM_MISSING,
    ERR_PARAM_RANGE,
    ERR_PARSE,
    SkillError,
)


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise SkillError(ERR_PARAM_RANGE, "exponent magnitude must not exceed 12", {"limit": 12})

        result = _BINARY_OPERATORS[type(node.op)](left, right)

        if isinstance(result, complex) or not math.isfinite(float(result)) or abs(result) > 1e100:
            raise SkillError(ERR_CALCULATION, "calculation result is out of range")
        return result

    raise SkillError(ERR_PARSE, f"unsupported expression element: {type(node).__name__}", {"node": type(node).__name__})


def calculator(expression: str) -> dict:
    if not isinstance(expression, str) or not expression.strip():
        raise SkillError(ERR_PARAM_MISSING, "expression must be a non-empty string", {"param": "expression"})
    if len(expression) > 200:
        raise SkillError(ERR_PARAM_RANGE, "expression is too long", {"param": "expression", "max_length": 200})

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise SkillError(ERR_PARSE, "invalid arithmetic expression", {"expression": expression[:50]}) from exc

    return {"result": _evaluate(tree)}