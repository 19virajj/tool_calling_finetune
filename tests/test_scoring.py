from helpers import CONVERT, WEATHER, call_text, make_eval_record

from toolft.scoring import aggregate, canon, score_example, wilson

GOLD_W = [{"name": "get_weather", "arguments": {"city": "Paris", "units": "celsius"}}]


def weather_example():
    return make_eval_record([WEATHER, CONVERT], GOLD_W)


def test_perfect_output():
    s = score_example(weather_example(), call_text(*GOLD_W))
    assert s["valid_call"] and s["tool_correct"] and s["args_correct"] and s["clean_format"]
    assert s["failure"] is None


def test_prose_around_call_is_valid_but_not_clean():
    text = "Sure, here you go:\n" + call_text(*GOLD_W)
    s = score_example(weather_example(), text)
    assert s["valid_call"] and s["args_correct"]
    assert not s["clean_format"]


def test_wrong_argument_value_is_valid_but_not_correct():
    wrong = [{"name": "get_weather", "arguments": {"city": "London", "units": "celsius"}}]
    s = score_example(weather_example(), call_text(*wrong))
    assert s["valid_call"] and s["tool_correct"]
    assert not s["args_correct"]


def test_wrong_tool():
    other = [{"name": "convert_currency", "arguments": {"amount": 1, "from_cur": "USD", "to_cur": "EUR"}}]
    s = score_example(weather_example(), call_text(*other))
    assert s["valid_call"]
    assert not s["tool_correct"] and not s["args_correct"]


def test_missing_required_is_invalid():
    bad = [{"name": "get_weather", "arguments": {"units": "celsius"}}]
    s = score_example(weather_example(), call_text(*bad))
    assert not s["valid_call"]
    assert s["failure"] == "missing_required"


def test_malformed_json_and_no_call():
    s = score_example(weather_example(), "<tool_call>\n{broken\n</tool_call>")
    assert not s["valid_call"] and s["failure"] == "parse_error"
    s = score_example(weather_example(), "I cannot help with that.")
    assert not s["valid_call"] and s["failure"] == "no_call"


def test_one_bad_call_among_good_makes_output_invalid():
    ex = make_eval_record(
        [WEATHER],
        [GOLD_W[0], {"name": "get_weather", "arguments": {"city": "Rome", "units": "celsius"}}],
    )
    out = call_text(GOLD_W[0], {"name": "get_weather", "arguments": {"units": "celsius"}})
    s = score_example(ex, out)
    assert not s["valid_call"]


def test_multiple_calls_order_insensitive():
    g2 = {"name": "get_weather", "arguments": {"city": "Rome", "units": "celsius"}}
    ex = make_eval_record([WEATHER], [GOLD_W[0], g2])
    s = score_example(ex, call_text(g2, GOLD_W[0]))
    assert s["args_correct"]
    s = score_example(ex, call_text(GOLD_W[0]))  # missing the second call
    assert s["valid_call"] and not s["tool_correct"]


def test_numeric_normalisation():
    assert canon(1.0) == canon(1)
    assert canon({"b": 2.0, "a": [1.0]}) == {"a": [1], "b": 2}
    assert canon(True) is True  # bool must not collapse into 1
    gold = [{"name": "convert_currency", "arguments": {"amount": 10, "from_cur": "USD", "to_cur": "EUR"}}]
    ex = make_eval_record([CONVERT], gold)
    out = [{"name": "convert_currency", "arguments": {"amount": 10.0, "from_cur": "USD", "to_cur": "EUR"}}]
    assert score_example(ex, call_text(*out))["args_correct"]


def test_abstain_slice():
    ex = make_eval_record([CONVERT], [], slice_="abstain")
    ok = score_example(ex, "None of the available tools can handle this request.<|im_end|>")
    assert ok["abstain_correct"]
    bad = score_example(ex, call_text({"name": "convert_currency", "arguments": {}}))
    assert not bad["abstain_correct"] and bad["failure"] == "called_tool_when_none_fits"
    broken = score_example(ex, "<tool_call>\n{oops")  # attempted a call: not an abstention
    assert not broken["abstain_correct"]


def test_wilson_interval():
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(94, 100)
    assert lo < 0.94 < hi
    assert 0.86 < lo < 0.89 and 0.96 < hi < 0.99
    lo, hi = wilson(500, 1000)
    assert abs((lo + hi) / 2 - 0.5) < 0.005
    # more data, tighter interval
    small = wilson(94, 100)
    large = wilson(940, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_aggregate():
    good = score_example(weather_example(), call_text(*GOLD_W))
    bad = score_example(weather_example(), "no call here")
    ab = score_example(make_eval_record([CONVERT], [], slice_="abstain"), "No tool fits.")
    m = aggregate([good, good, bad, ab])
    assert m["n_call"] == 3 and m["n_abstain"] == 1
    assert m["valid_call"]["k"] == 2 and m["valid_call"]["n"] == 3
    assert m["abstain_correct"]["rate"] == 1.0
    assert m["failures"] == {"no_call": 1}
