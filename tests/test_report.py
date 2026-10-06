import tempfile
from pathlib import Path

from helpers import WEATHER, call_text, make_eval_record

import make_report as mr
from toolft.scoring import aggregate, score_example

GOLD = [{"name": "get_weather", "arguments": {"city": "Paris", "units": "celsius"}}]


def fake_metrics(name: str, n_good: int, n_bad: int) -> dict:
    ex = make_eval_record([WEATHER], GOLD)
    results = [score_example(ex, call_text(*GOLD)) for _ in range(n_good)]
    results += [score_example(ex, "no call") for _ in range(n_bad)]
    ab = make_eval_record([WEATHER], [], slice_="abstain")
    results += [score_example(ab, "No tool fits.") for _ in range(5)]
    m = aggregate(results)
    m["run"] = {"name": name, "model": "m", "decoding": "greedy", "max_new_tokens": 256}
    return m


TARGETS = {"baseline_valid_call_rate": 0.71, "tuned_valid_call_rate": 0.94}


def test_report_contains_measured_values_and_goal_status():
    runs = {"base": fake_metrics("base", 70, 30), "tuned": fake_metrics("tuned", 95, 5)}
    md = mr.build_markdown(runs, TARGETS)
    assert "70.0%" in md and "95.0%" in md
    assert "Goal set before running" in md
    assert "targets, not results" in md
    assert "Tuned target (94%): met" in md
    assert "+25.0 points" in md


def test_report_says_not_met_when_below_target():
    runs = {"base": fake_metrics("base", 70, 30), "tuned": fake_metrics("tuned", 80, 20)}
    assert "not met" in mr.build_markdown(runs, TARGETS)


def test_report_handles_missing_runs():
    assert "No results found" in mr.build_markdown({}, TARGETS)
    md = mr.build_markdown({"base": fake_metrics("base", 7, 3)}, TARGETS)
    assert "Tuned target" not in md


def test_chart_is_written():
    runs = {"base": fake_metrics("base", 70, 30), "tuned": fake_metrics("tuned", 95, 5)}
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "c.png"
        mr.make_chart(runs, out)
        assert out.exists() and out.stat().st_size > 5000
