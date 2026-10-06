#!/usr/bin/env python
"""Build train.jsonl / eval.jsonl from a function-calling dataset (default: xlam-60k).

Design choices that make the final number defensible:

* Held-out TOOLS, not just held-out rows. Tool names are hashed into train / test buckets.
  A training example may not contain any test-bucket tool anywhere in its tool list; an eval
  example's gold calls use only test-bucket tools. So the eval measures generalisation to
  tools never seen in training.
* Gold labels are validated against the same schema validator used for scoring. Rows whose
  gold answer fails the validator are dropped, so a "valid" score of 100% is reachable.
* An "abstain" slice: the request is paired with a tool list that does not contain the right
  tool, and the correct behaviour is to NOT call anything. Training includes a small share of
  these so the model does not learn to always emit a call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toolft.io import load_config, read_jsonl, write_json, write_jsonl  # noqa: E402
from toolft.schema import normalize_tool, validate_call  # noqa: E402

ABSTAIN_TEXT = "None of the available tools can handle this request."


def _loads(x):
    return json.loads(x) if isinstance(x, str) else x


def build_example(row: dict) -> tuple[dict | None, str | None]:
    """Convert one raw dataset row to an internal example, or (None, reason) if unusable."""
    try:
        raw_tools = _loads(row["tools"])
        answers = _loads(row["answers"])
        query = row["query"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return None, "malformed_row"
    if (
        not isinstance(raw_tools, list)
        or not raw_tools
        or not isinstance(answers, list)
        or not answers
        or not isinstance(query, str)
        or not query.strip()
    ):
        return None, "malformed_row"

    try:
        tools = [normalize_tool(t) for t in raw_tools]
    except (KeyError, AttributeError, TypeError):
        return None, "bad_tool_schema"

    gold = []
    for a in answers:
        if not isinstance(a, dict) or not isinstance(a.get("arguments"), dict):
            return None, "malformed_row"
        if validate_call(a, tools):
            return None, "gold_fails_schema"
        gold.append({"name": a["name"], "arguments": a["arguments"]})

    return {"query": query.strip(), "tools": tools, "gold_calls": gold}, None


def tool_names(ex: dict) -> set[str]:
    return {t["function"]["name"] for t in ex["tools"]}


def gold_names(ex: dict) -> set[str]:
    return {g["name"] for g in ex["gold_calls"]}


def is_test_name(name: str, pct: int, seed: int) -> bool:
    h = int(hashlib.md5(f"{seed}:{name}".encode()).hexdigest(), 16)
    return (h % 100) < pct


def split_pools(examples: list[dict], pct: int, seed: int) -> tuple[list[dict], list[dict], int]:
    """Return (train_pool, test_pool, n_mixed_dropped)."""
    train, test, mixed = [], [], 0
    for ex in examples:
        in_test = {n for n in tool_names(ex) if is_test_name(n, pct, seed)}
        if not in_test:
            train.append(ex)
        elif gold_names(ex) <= in_test:
            test.append(ex)
        else:
            mixed += 1  # touches test tools but is not a clean held-out example
    return train, test, mixed


def to_record(ex: dict, rec_id: str) -> dict:
    calls = [
        {"type": "function", "function": {"name": g["name"], "arguments": g["arguments"]}}
        for g in ex["gold_calls"]
    ]
    return {
        "id": rec_id,
        "slice": "call",
        "tools": ex["tools"],
        "messages": [
            {"role": "user", "content": ex["query"]},
            {"role": "assistant", "content": "", "tool_calls": calls},
        ],
        "gold_calls": ex["gold_calls"],
    }


def make_abstain(ex: dict, pool: list[dict], rng: random.Random, rec_id: str) -> dict | None:
    """Pair the query with another example's tools, none of which is a gold tool of this query."""
    wanted = gold_names(ex)
    for _ in range(25):
        donor = rng.choice(pool)
        if donor is ex:
            continue
        if wanted.isdisjoint(tool_names(donor)):
            return {
                "id": rec_id,
                "slice": "abstain",
                "tools": donor["tools"],
                "messages": [
                    {"role": "user", "content": ex["query"]},
                    {"role": "assistant", "content": ABSTAIN_TEXT},
                ],
                "gold_calls": [],
            }
    return None


def build_splits(examples: list[dict], cfg: dict, seed: int) -> tuple[list[dict], list[dict], dict]:
    d = cfg["data"]
    rng = random.Random(seed)
    train_pool, test_pool, mixed = split_pools(examples, d["test_tool_pct"], seed)
    for pool in (train_pool, test_pool):
        pool.sort(key=lambda e: e["query"])
        rng.shuffle(pool)

    train_calls = train_pool[: d["max_train"]]
    eval_calls = test_pool[: d["max_eval"]]

    train_recs = [to_record(e, f"train-{i}") for i, e in enumerate(train_calls)]
    n_abs_train = int(round(d["abstain_train_frac"] * len(train_calls)))
    for i, e in enumerate(rng.sample(train_calls, min(n_abs_train, len(train_calls)))):
        rec = make_abstain(e, train_pool, rng, f"train-abstain-{i}")
        if rec:
            train_recs.append(rec)
    rng.shuffle(train_recs)

    eval_recs = [to_record(e, f"eval-{i}") for i, e in enumerate(eval_calls)]
    n_abs_eval = min(d["abstain_eval"], len(eval_calls))
    for i, e in enumerate(rng.sample(eval_calls, n_abs_eval)):
        rec = make_abstain(e, test_pool, rng, f"eval-abstain-{i}")
        if rec:
            eval_recs.append(rec)

    stats = {
        "usable_examples": len(examples),
        "train_pool": len(train_pool),
        "test_pool": len(test_pool),
        "dropped_mixed_test_tools": mixed,
        "train_records": len(train_recs),
        "train_call": sum(r["slice"] == "call" for r in train_recs),
        "train_abstain": sum(r["slice"] == "abstain" for r in train_recs),
        "eval_records": len(eval_recs),
        "eval_call": sum(r["slice"] == "call" for r in eval_recs),
        "eval_abstain": sum(r["slice"] == "abstain" for r in eval_recs),
        "test_tool_pct": d["test_tool_pct"],
    }
    return train_recs, eval_recs, stats


def load_rows(cfg: dict, input_jsonl: str | None):
    if input_jsonl:
        return read_jsonl(input_jsonl)
    from datasets import load_dataset

    return load_dataset(cfg["data"]["dataset"], split="train")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--input-jsonl", help="local JSONL with query/tools/answers instead of downloading")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]

    examples, dropped, seen = [], Counter(), set()
    for row in load_rows(cfg, args.input_jsonl):
        ex, reason = build_example(row)
        if ex is None:
            dropped[reason] += 1
            continue
        if ex["query"] in seen:
            dropped["duplicate_query"] += 1
            continue
        seen.add(ex["query"])
        examples.append(ex)

    train_recs, eval_recs, stats = build_splits(examples, cfg, seed)
    stats["dropped"] = dict(dropped)

    out = Path(cfg["data"]["out_dir"])
    write_jsonl(out / "train.jsonl", train_recs)
    write_jsonl(out / "eval.jsonl", eval_recs)
    write_json(out / "stats.json", stats)
    print(json.dumps(stats, indent=2))

    if stats["eval_call"] < 200:
        print("WARNING: fewer than 200 eval call examples; confidence intervals will be wide.", file=sys.stderr)


if __name__ == "__main__":
    main()
