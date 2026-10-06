import json

from helpers import CONVERT, WEATHER, xlam_row

import prepare_data as pd
from toolft.schema import validate_call


def make_row(i: int, n_distractors: int = 2) -> dict:
    """One synthetic xlam-style row whose gold tool is tool_<i>, with distractor tools."""
    def tool(k):
        return {
            "name": f"tool_{k}",
            "description": f"Tool number {k}",
            "parameters": {"q": {"type": "str", "description": "query"}, "n": {"type": "int", "description": "count", "default": 1}},
        }

    defs = [tool(i)] + [tool((i * 7 + d + 1) % 400) for d in range(n_distractors)]
    # de-duplicate by name, keep order
    seen, uniq = set(), []
    for d in defs:
        if d["name"] not in seen:
            seen.add(d["name"])
            uniq.append(d)
    return xlam_row(f"please run tool {i}", uniq, [{"name": f"tool_{i}", "arguments": {"q": f"item {i}"}}])


def examples(n: int = 400) -> list[dict]:
    out = []
    for i in range(n):
        ex, reason = pd.build_example(make_row(i))
        assert reason is None, reason
        out.append(ex)
    return out


def cfg(**over) -> dict:
    d = {
        "test_tool_pct": 15,
        "max_train": 200,
        "max_eval": 40,
        "abstain_train_frac": 0.1,
        "abstain_eval": 10,
    }
    d.update(over)
    return {"data": d}


def test_build_example_ok_and_normalised():
    ex, reason = pd.build_example(xlam_row("weather in Paris", [WEATHER, CONVERT], [{"name": "get_weather", "arguments": {"city": "Paris"}}]))
    assert reason is None
    assert ex["gold_calls"] == [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    assert ex["tools"][0]["type"] == "function"
    assert validate_call(ex["gold_calls"][0], ex["tools"]) == []


def test_build_example_accepts_already_parsed_json():
    row = {"query": "q", "tools": [WEATHER], "answers": [{"name": "get_weather", "arguments": {"city": "X"}}]}
    ex, reason = pd.build_example(row)
    assert reason is None and ex is not None


def test_build_example_rejects_bad_rows():
    # gold call missing a required argument
    _, reason = pd.build_example(xlam_row("q", [WEATHER], [{"name": "get_weather", "arguments": {}}]))
    assert reason == "gold_fails_schema"
    # gold call to a tool that is not offered
    _, reason = pd.build_example(xlam_row("q", [WEATHER], [{"name": "convert_currency", "arguments": {}}]))
    assert reason == "gold_fails_schema"
    # broken JSON / missing fields / empty query
    assert pd.build_example({"query": "q", "tools": "{oops", "answers": "[]"})[1] == "malformed_row"
    assert pd.build_example({"tools": "[]", "answers": "[]"})[1] == "malformed_row"
    assert pd.build_example(xlam_row("  ", [WEATHER], [{"name": "get_weather", "arguments": {"city": "X"}}]))[1] == "malformed_row"
    # tool without a name
    assert pd.build_example(xlam_row("q", [{"description": "x", "parameters": {}}], [{"name": "a", "arguments": {}}]))[1] == "bad_tool_schema"


def test_split_has_no_tool_leakage():
    exs = examples()
    train, test, mixed = pd.split_pools(exs, 15, seed=1)
    assert len(train) > 50 and len(test) > 10
    assert mixed > 0  # some examples touch test tools only as distractors

    test_gold = set().union(*(pd.gold_names(e) for e in test))
    train_tools = set().union(*(pd.tool_names(e) for e in train))
    assert test_gold.isdisjoint(train_tools)  # held-out tools never appear in training prompts
    for e in test:
        assert all(pd.is_test_name(n, 15, 1) for n in pd.gold_names(e))
    for e in train:
        assert not any(pd.is_test_name(n, 15, 1) for n in pd.tool_names(e))


def test_split_is_deterministic():
    a = pd.split_pools(examples(), 15, seed=1)
    b = pd.split_pools(examples(), 15, seed=1)
    assert [e["query"] for e in a[1]] == [e["query"] for e in b[1]]
    c = pd.split_pools(examples(), 15, seed=2)
    assert [e["query"] for e in a[1]] != [e["query"] for e in c[1]]


def test_build_splits_records():
    train, ev, stats = pd.build_splits(examples(), cfg(), seed=1)
    assert stats["train_call"] > 0 and stats["train_abstain"] > 0
    assert stats["eval_call"] <= 40 and stats["eval_abstain"] <= 10
    assert stats["eval_abstain"] > 0

    for rec in train + ev:
        assert rec["slice"] in ("call", "abstain")
        assert rec["messages"][0]["role"] == "user"
        assert rec["messages"][1]["role"] == "assistant"
        json.dumps(rec)  # serialisable

    call = next(r for r in ev if r["slice"] == "call")
    tc = call["messages"][1]["tool_calls"]
    assert tc[0]["function"]["name"] == call["gold_calls"][0]["name"]
    assert isinstance(tc[0]["function"]["arguments"], dict)  # dict, not a JSON string, for the chat template

    # eval records never share a query between slices' gold tools and the train tool lists
    train_tools = {t["function"]["name"] for r in train for t in r["tools"]}
    eval_gold = {g["name"] for r in ev if r["slice"] == "call" for g in r["gold_calls"]}
    assert eval_gold.isdisjoint(train_tools)


def test_abstain_records_do_not_offer_the_right_tool():
    _, ev, _ = pd.build_splits(examples(), cfg(), seed=1)
    originals = {}
    for r in ev:
        if r["slice"] == "call":
            originals[r["messages"][0]["content"]] = {g["name"] for g in r["gold_calls"]}
    n = 0
    for r in ev:
        if r["slice"] != "abstain":
            continue
        n += 1
        offered = {t["function"]["name"] for t in r["tools"]}
        assert originals[r["messages"][0]["content"]].isdisjoint(offered)
        assert r["gold_calls"] == []
        assert r["messages"][1]["content"] == pd.ABSTAIN_TEXT
    assert n > 0


def test_build_splits_is_reproducible():
    a = pd.build_splits(examples(), cfg(), seed=5)
    b = pd.build_splits(examples(), cfg(), seed=5)
    assert [r["id"] for r in a[0]] == [r["id"] for r in b[0]]
    assert [r["messages"][0]["content"] for r in a[1]] == [r["messages"][0]["content"] for r in b[1]]
