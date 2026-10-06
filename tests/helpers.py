"""Shared synthetic data for the tests (no network, no GPU)."""
from __future__ import annotations

import json

WEATHER = {
    "name": "get_weather",
    "description": "Get the weather for a city.",
    "parameters": {
        "city": {"type": "str", "description": "City name"},
        "units": {"type": "str", "description": "celsius or fahrenheit", "default": "celsius"},
    },
}

CONVERT = {
    "name": "convert_currency",
    "description": "Convert an amount between currencies.",
    "parameters": {
        "amount": {"type": "float", "description": "Amount"},
        "from_cur": {"type": "str", "description": "Source currency"},
        "to_cur": {"type": "str", "description": "Target currency"},
    },
}

LIST_TOOL = {
    "name": "add_tags",
    "description": "Add tags to an item.",
    "parameters": {
        "item_id": {"type": "int", "description": "Item id"},
        "tags": {"type": "List[str]", "description": "Tags"},
        "dry_run": {"type": "Optional[bool]", "description": "Do not write"},
    },
}


def xlam_row(query: str, tool_defs: list[dict], answers: list[dict]) -> dict:
    """A row shaped like xlam-function-calling-60k (tools and answers are JSON strings)."""
    return {"query": query, "tools": json.dumps(tool_defs), "answers": json.dumps(answers)}


def make_eval_record(tool_defs: list[dict], gold: list[dict], slice_: str = "call") -> dict:
    from toolft.schema import normalize_tool

    return {
        "id": "t-0",
        "slice": slice_,
        "tools": [normalize_tool(t) for t in tool_defs],
        "gold_calls": gold if slice_ == "call" else [],
        "messages": [{"role": "user", "content": "x"}],
    }


def call_text(*calls: dict) -> str:
    """Render calls the way Qwen emits them."""
    return "\n".join(f"<tool_call>\n{json.dumps(c)}\n</tool_call>" for c in calls) + "<|im_end|>"
