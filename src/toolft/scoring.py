"""Scoring of model outputs against gold tool calls.

Metrics on the "call" slice (the request should produce one or more tool calls):

  valid_call    : output parses, has at least one call, and every call is schema-valid
                  (known tool, all required args, no unknown args, correct JSON types).
  tool_correct  : the multiset of tool names equals the gold multiset.
  args_correct  : tool_correct and every call's arguments equal the gold arguments.
  parse_ok      : every <tool_call> block is valid JSON.
  clean_format  : nothing but <tool_call> blocks in the output (no surrounding prose).

Metric on the "abstain" slice (no provided tool fits the request):

  abstain_correct : the output contains no <tool_call> at all.

valid_call is a structural metric; args_correct is the strict one. Report both.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

from .parsing import parse_output
from .schema import validate_call


def canon(value: Any) -> Any:
    """Canonical form for comparing argument values (1.0 == 1, key order ignored)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: canon(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [canon(v) for v in value]
    return value


def call_key(call: Any) -> str:
    if not isinstance(call, dict):
        return json.dumps({"invalid": repr(call)})
    return json.dumps(
        {"name": call.get("name"), "arguments": canon(call.get("arguments", {}))},
        sort_keys=True,
        ensure_ascii=False,
    )


def _name(call: Any) -> str:
    return str(call.get("name")) if isinstance(call, dict) else ""


def score_example(example: dict, text: str) -> dict:
    """Score one model output. `example` is a record from eval.jsonl."""
    parsed = parse_output(text)
    result: dict[str, Any] = {
        "slice": example["slice"],
        "n_calls": len(parsed.calls),
        "parse_ok": parsed.n_parse_errors == 0,
        "clean_format": parsed.extra_text == "",
        "failure": None,
    }

    if example["slice"] == "abstain":
        result["abstain_correct"] = parsed.n_blocks == 0
        if not result["abstain_correct"]:
            result["failure"] = "called_tool_when_none_fits"
        return result

    tools = example["tools"]
    gold = example["gold_calls"]
    problems = [validate_call(c, tools) for c in parsed.calls]
    has_call = len(parsed.calls) > 0

    result["valid_call"] = result["parse_ok"] and has_call and all(not p for p in problems)
    result["tool_correct"] = sorted(_name(c) for c in parsed.calls) == sorted(g["name"] for g in gold)
    result["args_correct"] = result["tool_correct"] and (
        sorted(call_key(c) for c in parsed.calls) == sorted(call_key(g) for g in gold)
    )

    if not result["valid_call"]:
        if not result["parse_ok"]:
            result["failure"] = "parse_error"
        elif not has_call:
            result["failure"] = "no_call"
        else:
            first = next(p[0] for p in problems if p)
            result["failure"] = first.split(":", 1)[0]
    return result


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rate(flags: list[bool]) -> dict:
    n = len(flags)
    k = sum(1 for f in flags if f)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": [lo, hi]}


def aggregate(results: list[dict]) -> dict:
    calls = [r for r in results if r["slice"] == "call"]
    abstain = [r for r in results if r["slice"] == "abstain"]
    out: dict[str, Any] = {"n_call": len(calls), "n_abstain": len(abstain)}
    for key in ("valid_call", "tool_correct", "args_correct", "parse_ok", "clean_format"):
        out[key] = _rate([r[key] for r in calls])
    out["abstain_correct"] = _rate([r["abstain_correct"] for r in abstain])
    out["failures"] = dict(Counter(r["failure"] for r in results if r["failure"]))
    return out
