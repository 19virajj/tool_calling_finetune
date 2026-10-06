from helpers import CONVERT, LIST_TOOL, WEATHER

from toolft.parsing import parse_output
from toolft.schema import json_type, normalize_tool, validate_call


def tools(*defs):
    return [normalize_tool(d) for d in defs]


def test_json_type_mapping():
    assert json_type("int") == "integer"
    assert json_type("float") == "number"
    assert json_type("str") == "string"
    assert json_type("bool") == "boolean"
    assert json_type("List[str]") == "array"
    assert json_type("Optional[List[int]]") == "array"
    assert json_type("int, optional") == "integer"
    assert json_type("Dict") == "object"
    assert json_type("SomethingWeird") is None
    assert json_type(None) is None


def test_normalize_required_and_optional():
    fn = normalize_tool(WEATHER)["function"]
    assert fn["name"] == "get_weather"
    assert fn["parameters"]["required"] == ["city"]  # units has a default
    assert fn["parameters"]["properties"]["units"]["default"] == "celsius"

    fn = normalize_tool(LIST_TOOL)["function"]
    assert fn["parameters"]["required"] == ["item_id", "tags"]  # dry_run is Optional[...]
    assert fn["parameters"]["properties"]["tags"]["type"] == "array"


def test_normalize_passes_through_json_schema_tools():
    raw = {
        "name": "f",
        "description": "d",
        "parameters": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
    }
    fn = normalize_tool(raw)["function"]
    assert fn["parameters"]["required"] == ["a"]
    assert fn["parameters"]["properties"]["a"]["type"] == "string"


def test_valid_call_passes():
    t = tools(WEATHER)
    assert validate_call({"name": "get_weather", "arguments": {"city": "Paris"}}, t) == []
    assert validate_call({"name": "get_weather", "arguments": {"city": "Paris", "units": "celsius"}}, t) == []


def test_missing_required():
    errs = validate_call({"name": "get_weather", "arguments": {"units": "celsius"}}, tools(WEATHER))
    assert errs == ["missing_required: city"]


def test_unknown_argument():
    errs = validate_call({"name": "get_weather", "arguments": {"city": "Paris", "zip": "75001"}}, tools(WEATHER))
    assert errs == ["unknown_arg: zip"]


def test_wrong_types():
    t = tools(LIST_TOOL)
    assert validate_call({"name": "add_tags", "arguments": {"item_id": 3, "tags": ["a"]}}, t) == []
    assert validate_call({"name": "add_tags", "arguments": {"item_id": "3", "tags": ["a"]}}, t)[0].startswith("wrong_type")
    assert validate_call({"name": "add_tags", "arguments": {"item_id": True, "tags": ["a"]}}, t)[0].startswith("wrong_type")
    assert validate_call({"name": "add_tags", "arguments": {"item_id": 3, "tags": "a"}}, t)[0].startswith("wrong_type")
    assert validate_call({"name": "add_tags", "arguments": {"item_id": 3, "tags": [], "dry_run": "no"}}, t)[0].startswith("wrong_type")


def test_number_accepts_int_and_float():
    t = tools(CONVERT)
    for amount in (10, 10.5):
        call = {"name": "convert_currency", "arguments": {"amount": amount, "from_cur": "USD", "to_cur": "EUR"}}
        assert validate_call(call, t) == []


def test_unknown_tool_and_bad_shapes():
    t = tools(WEATHER)
    assert validate_call({"name": "nope", "arguments": {}}, t)[0].startswith("unknown_tool")
    assert validate_call({"name": "get_weather", "arguments": "{}"}, t)[0].startswith("bad_arguments")
    assert validate_call({"arguments": {}}, t) == ["no_name"]
    assert validate_call(["x"], t) == ["not_object"]


def test_parse_output_variants():
    ok = parse_output('<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call><|im_end|>')
    assert len(ok.calls) == 1 and ok.n_parse_errors == 0 and ok.extra_text == ""

    two = parse_output(
        '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n<tool_call>\n{"name": "b", "arguments": {}}\n</tool_call>'
    )
    assert [c["name"] for c in two.calls] == ["a", "b"]

    bad = parse_output("<tool_call>\n{not json}\n</tool_call>")
    assert bad.calls == [] and bad.n_parse_errors == 1 and bad.n_blocks == 1

    cut = parse_output('<tool_call>\n{"name": "a", "arguments": {"x"')
    assert cut.calls == [] and cut.n_parse_errors == 1

    prose = parse_output('Sure! <tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>')
    assert prose.extra_text == "Sure!" and len(prose.calls) == 1

    none = parse_output("None of the tools fit.<|im_end|><|endoftext|><|endoftext|>")
    assert none.n_blocks == 0 and none.extra_text == "None of the tools fit."
