"""Extract tool calls from raw model output (Qwen / Hermes `<tool_call>` format)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_SPECIAL = re.compile(r"<\|(?:im_end|im_start|endoftext)\|>")


@dataclass
class ParsedOutput:
    calls: list = field(default_factory=list)  # JSON values that parsed (should be dicts)
    n_blocks: int = 0  # number of "<tool_call>" openers seen, parsed or not
    n_parse_errors: int = 0  # blocks that were not valid JSON, or never closed
    extra_text: str = ""  # any text outside tool_call blocks


def parse_output(text: str) -> ParsedOutput:
    text = _SPECIAL.sub("", text)
    out = ParsedOutput(n_blocks=text.count("<tool_call>"))

    for body in _BLOCK.findall(text):
        try:
            out.calls.append(json.loads(body))
        except json.JSONDecodeError:
            out.n_parse_errors += 1

    remainder = _BLOCK.sub("", text)
    unterminated = remainder.count("<tool_call>")
    if unterminated:
        out.n_parse_errors += unterminated
        remainder = remainder.replace("<tool_call>", "")
    out.extra_text = remainder.strip()
    return out
