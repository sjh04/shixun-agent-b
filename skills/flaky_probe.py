from __future__ import annotations


class RecoverableToolError(RuntimeError):
    """A transient error that B3 is allowed to retry (进阶 2)."""


# Module-level attempt counter so repeated retries within one B3 call progress
# deterministically. Reset to 0 once the call finally succeeds.
_ATTEMPTS = 0


def flaky_probe(fail_times: int) -> dict:
    """Fail with a recoverable error for the first ``fail_times`` attempts.

    Used only to demonstrate B3's limited-retry behaviour; not a business skill.
    """
    global _ATTEMPTS
    if not isinstance(fail_times, int) or isinstance(fail_times, bool) or fail_times < 0:
        raise ValueError("fail_times must be a non-negative integer")
    _ATTEMPTS += 1
    if _ATTEMPTS <= fail_times:
        raise RecoverableToolError(
            f"transient failure on attempt {_ATTEMPTS} (need {fail_times + 1})"
        )
    succeeded_on = _ATTEMPTS
    _ATTEMPTS = 0
    return {"attempt": succeeded_on, "message": "flaky_probe succeeded"}
