"""Tool-schema normalisation and tool-call validation.

xlam-style datasets describe parameters as {"name": {"type": "int", "description": ...}}.
We convert them to standard JSON-Schema function tools so that (a) the Qwen chat
template can render them and (b) one validator serves the training data, the gold
labels and the model output.
"""
from __future__ import annotations

import re
from typing import Any

_SCALARS = {
    "int": "integer",
    "integer": "integer",
    "long": "integer",
    "float": "number",
    "double": "number",
    "number": "number",
    "str": "string",
    "string": "string",
    "text": "string",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "tuple": "array",
    "set": "array",
    "dict": "object",
    "object": "object",
    "map": "object",
}


def json_type(type_str: Any) -> str | None:
    """Map a Python-ish type string ("int", "List[str]", "Optional[float]") to a JSON type.

    Returns None for types we cannot classify; those parameters are not type-checked.
    """
    if not isinstance(type_str, str):
        return None
    t = type_str.strip()
    m = re.fullmatch(r"(?i)optional\[(.+)\]", t)
    if m:
        t = m.group(1).strip()
    base = re.split(r"[\[\(,\s]", t, maxsplit=1)[0].lower()
    return _SCALARS.get(base)


def is_optional(spec: dict) -> bool:
    """A parameter is optional if it has a default or its type says so."""
    return "default" in spec or "optional" in str(spec.get("type", "")).lower()


def normalize_tool(raw: dict) -> dict:
    """Convert one raw tool description to an OpenAI/JSON-Schema style function tool.

    Raises KeyError / TypeError / AttributeError on malformed input; callers drop those rows.
    """
    name = raw["name"]
    if not isinstance(name, str) or not name:
        raise KeyError("name")
    params = raw.get("parameters") or {}

    # Already JSON-Schema shaped: pass through.
    if isinstance(params, dict) and params.get("type") == "object" and "properties" in params:
        parameters = {
            "type": "object",
            "properties": params["properties"],
            "required": list(params.get("required", [])),
        }
    else:
        props: dict[str, dict] = {}
        required: list[str] = []
        for pname, spec in params.items():
            if not isinstance(spec, dict):
                spec = {"type": str(spec)}
            prop: dict[str, Any] = {}
            jt = json_type(spec.get("type"))
            if jt:
                prop["type"] = jt
            if spec.get("description"):
                prop["description"] = str(spec["description"])
            if "default" in spec:
                prop["default"] = spec["default"]
            props[pname] = prop
            if not is_optional(spec):
                required.append(pname)
        parameters = {"type": "object", "properties": props, "required": required}

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": str(raw.get("description", "")),
            "parameters": parameters,
        },
    }


def tool_index(tools: list[dict]) -> dict[str, dict]:
    return {t["function"]["name"]: t for t in tools}


def _type_ok(value: Any, jtype: str) -> bool:
    if jtype == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if jtype == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if jtype == "string":
        return isinstance(value, str)
    if jtype == "boolean":
        return isinstance(value, bool)
    if jtype == "array":
        return isinstance(value, list)
    if jtype == "object":
        return isinstance(value, dict)
    return True


def validate_call(call: Any, tools: list[dict]) -> list[str]:
    """Return a list of problems with one call; an empty list means the call is valid.

    Problem strings start with a stable prefix (before the colon) so they can be counted:
    not_object, no_name, unknown_tool, bad_arguments, missing_required, unknown_arg, wrong_type.
    """
    if not isinstance(call, dict):
        return ["not_object"]
    name = call.get("name")
    if not isinstance(name, str):
        return ["no_name"]
    spec = tool_index(tools).get(name)
    if spec is None:
        return [f"unknown_tool: {name}"]
    args = call.get("arguments")
    if not isinstance(args, dict):
        return ["bad_arguments: arguments is not a JSON object"]

    params = spec["function"]["parameters"]
    props = params.get("properties", {})
    errors: list[str] = []
    for req in params.get("required", []):
        if req not in args:
            errors.append(f"missing_required: {req}")
    for key, value in args.items():
        if key not in props:
            errors.append(f"unknown_arg: {key}")
            continue
        jtype = props[key].get("type")
        if jtype and not _type_ok(value, jtype):
            errors.append(f"wrong_type: {key} expected {jtype}")
    return errors
