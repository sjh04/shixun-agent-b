"""B2 Skill 统一错误码体系。

错误码为 4 位数字，千位表示错误大类：
    1xxx — 参数错误（缺失、类型不对、取值越界）
    2xxx — 文件错误（不存在、越权、格式不支持）
    3xxx — 执行错误（计算溢出、语法错误、超时）
    4xxx — 格式/转换错误
    5xxx — 复合 Skill 错误
    9xxx — 内部/框架错误
"""

from __future__ import annotations

ERR_PARAM_MISSING = 1001
ERR_PARAM_TYPE = 1002
ERR_PARAM_RANGE = 1003
ERR_PARAM_UNSUPPORTED = 1004

ERR_FILE_NOT_FOUND = 2001
ERR_FILE_ESCAPE = 2002
ERR_FILE_TYPE = 2003
ERR_FILE_READ = 2004

ERR_CALCULATION = 3001
ERR_PARSE = 3002
ERR_TIMEOUT = 3003
ERR_SANDBOX = 3004

ERR_FORMAT_INVALID = 4001
ERR_CONVERSION = 4002

ERR_PIPELINE_UNKNOWN_SKILL = 5001
ERR_PIPELINE_MISMATCH = 5002

ERR_UNKNOWN_SKILL = 9001
ERR_IMPORT = 9002
ERR_INTERNAL = 9999


class SkillError(Exception):
    """携带统一错误码的 Skill 异常。

    Attributes
    ----------
    code : int
        4 位错误码，见本模块常量。
    message : str
        面向开发者 / 日志的错误摘要。
    detail : dict | None
        可选的附加信息（如参数名、文件路径）。
    """

    def __init__(self, code: int, message: str, detail: dict | None = None) -> None:
        self.code = code
        self.message = message
        self.detail = detail or {}
        super().__init__(message)

    def as_error_dict(self) -> dict:
        """转为 SkillResult.error 所需的 JSON 结构。

        只保留三个核心字段：
          - code    : 错误码（程序判断用）
          - message : 错误描述（人看）
          - detail  : 结构化附加信息（程序取值用，非空时才出现）
        """
        err: dict = {
            "code": self.code,
            "message": self.message,
        }
        if self.detail:
            err["detail"] = self.detail
        return err