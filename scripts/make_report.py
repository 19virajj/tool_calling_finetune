#!/usr/bin/env python
"""Turn results/*/metrics.json into results/REPORT.md and results/comparison.png.

The goals in configs/default.yaml (targets) are reported next to what was actually measured.
Nothing in the report is filled in by hand.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toolft.io import load_config  # noqa: E402

PREFERRED_ORDER = ["base", "base_fewshot", "tuned"]
LABELS = {"base": "Base (zero-shot)", "base_fewshot": "Base (few-shot prompt)", "tuned": "Fine-tuned (QLoRA)"}
METRICS = [
    ("valid_call", "Valid call"),
    ("tool_correct", "Right tool"),
    ("args_correct", "Right args"),
]


def load_runs(results_dir: Path) -> dict[str, dict]:
    runs = {}
    for p in sorted(results_dir.glob("*/metrics.json")):
        with open(p, encoding="utf-8") as f:
            runs[p.parent.name] = json.load(f)
    ordered = {k: runs[k] for k in PREFERRED_ORDER if k in runs}
    ordered.update({k: v for k, v in runs.items() if k not in ordered})
    return ordered


def pct(r: dict | None) -> str:
    if not r or r.get("rate") is None:
        return "n/a"
    return f"{100 * r['rate']:.1f}%"


def pct_ci(r: dict | None) -> str:
    if not r or r.get("rate") is None:
        return "n/a"
    lo, hi = r["ci95"]
    return f"{100 * r['rate']:.1f}% ({100 * lo:.1f}-{100 * hi:.1f})"


def build_markdown(runs: dict[str, dict], targets: dict) -> str:
    lines = ["# Tool-calling fine-tune: before / after", ""]
    if not runs:
        return "\n".join(lines + ["No results found. Run scripts/evaluate.py first.", ""])

    any_run = next(iter(runs.values()))
    lines += [
        f"Model: `{any_run['run']['model']}`. Decoding: {any_run['run']['decoding']}, "
        f"max {any_run['run']['max_new_tokens']} new tokens. "
        f"Call slice n = {any_run['n_call']}, abstain slice n = {any_run['n_abstain']}. "
        "Eval tools are held out: no eval tool appears anywhere in the training data.",
        "",
        "Values are the measured rate with the 95% Wilson interval in brackets.",
        "",
        "| Run | Valid call | Right tool | Right args | Clean format | Abstain correct |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in runs.items():
        lines.append(
            f"| {LABELS.get(name, name)} | {pct_ci(m['valid_call'])} | {pct_ci(m['tool_correct'])} | "
            f"{pct_ci(m['args_correct'])} | {pct_ci(m['clean_format'])} | {pct_ci(m['abstain_correct'])} |"
        )

    lines += [
        "",
        "Definitions: valid call = parses, at least one call, and every call has a known tool, all required "
        "arguments, no unknown arguments and correct JSON types. Right tool = tool names match the gold calls. "
        "Right args = tool and argument values match the gold calls exactly. Clean format = nothing but "
        "tool-call blocks in the output. Abstain correct = no tool call when no provided tool fits.",
        "",
    ]

    # Comparison against the goal that was written down before running.
    tuned, base, fewshot = runs.get("tuned"), runs.get("base"), runs.get("base_fewshot")
    lines += ["## Goal vs measured", ""]
    lines += [
        f"Goal set before running: valid-call rate {100 * targets['baseline_valid_call_rate']:.0f}% -> "
        f"{100 * targets['tuned_valid_call_rate']:.0f}%. These are targets, not results.",
        "",
    ]
    if base and tuned:
        d = 100 * (tuned["valid_call"]["rate"] - base["valid_call"]["rate"])
        lines.append(f"- Fine-tuned vs base (zero-shot): {pct(base['valid_call'])} -> {pct(tuned['valid_call'])} ({d:+.1f} points).")
    if fewshot and tuned:
        d = 100 * (tuned["valid_call"]["rate"] - fewshot["valid_call"]["rate"])
        lines.append(f"- Fine-tuned vs base with a few-shot prompt: {pct(fewshot['valid_call'])} -> {pct(tuned['valid_call'])} ({d:+.1f} points).")
    if tuned:
        met = tuned["valid_call"]["rate"] >= targets["tuned_valid_call_rate"]
        lines.append(f"- Tuned target ({100 * targets['tuned_valid_call_rate']:.0f}%): {'met' if met else 'not met'}.")
    if base:
        diff = 100 * (base["valid_call"]["rate"] - targets["baseline_valid_call_rate"])
        lines.append(
            f"- Measured base rate differs from the assumed {100 * targets['baseline_valid_call_rate']:.0f}% by {diff:+.1f} points."
        )
    lines.append("")

    # Where failures come from.
    lines += ["## Failure breakdown", "", "Count of examples per first failure category (call and abstain slices).", ""]
    cats = sorted({c for m in runs.values() for c in m.get("failures", {})})
    if cats:
        lines += ["| Run | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
        for name, m in runs.items():
            f = m.get("failures", {})
            lines.append(f"| {LABELS.get(name, name)} | " + " | ".join(str(f.get(c, 0)) for c in cats) + " |")
    else:
        lines.append("No failures recorded.")
    lines += ["", "![comparison](comparison.png)", ""]
    return "\n".join(lines)


def make_chart(runs: dict[str, dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg, fg = "#0b0b0b", "#e6e6e6"
    greens = ["#2e5a3a", "#6fcf8a", "#00e676", "#b9f6ca"]
    names = list(runs)
    n_runs = len(names)
    width = 0.8 / max(n_runs, 1)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    for i, name in enumerate(names):
        vals, errs = [], [[], []]
        for key, _ in METRICS:
            r = runs[name][key]
            v = r["rate"] or 0.0
            lo, hi = r["ci95"]
            vals.append(100 * v)
            errs[0].append(100 * (v - lo))
            errs[1].append(100 * (hi - v))
        xs = [j + (i - (n_runs - 1) / 2) * width for j in range(len(METRICS))]
        ax.bar(xs, vals, width * 0.92, yerr=errs, color=greens[i % len(greens)], ecolor=fg,
               error_kw={"elinewidth": 1, "capsize": 3}, label=LABELS.get(name, name))
        for x, v, up in zip(xs, vals, errs[1]):
            ax.text(x, v + up + 1.5, f"{v:.0f}", ha="center", va="bottom", color=fg, fontsize=8)  # above the CI cap
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([label for _, label in METRICS], color=fg)
    ax.set_ylim(0, 112)
    ax.set_ylabel("% of held-out-tool examples", color=fg)
    ax.tick_params(colors=fg)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#555555")
    ax.yaxis.grid(True, color="#222222")
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, loc="upper center", ncol=n_runs, bbox_to_anchor=(0.5, 1.12))
    for t in leg.get_texts():
        t.set_color(fg)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=bg)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    results_dir = Path(cfg["eval"]["results_dir"])

    runs = load_runs(results_dir)
    md = build_markdown(runs, cfg["targets"])
    (results_dir / "REPORT.md").write_text(md, encoding="utf-8")
    if runs:
        make_chart(runs, results_dir / "comparison.png")
    print(f"Wrote {results_dir / 'REPORT.md'}" + (" and comparison.png" if runs else ""))


if __name__ == "__main__":
    main()
