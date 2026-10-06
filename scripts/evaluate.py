#!/usr/bin/env python
"""Run a model (optionally + LoRA adapter) over eval.jsonl and score the tool calls.

Examples:
  python scripts/evaluate.py --name base
  python scripts/evaluate.py --name base_fewshot --prompt-style fewshot
  python scripts/evaluate.py --name tuned --adapter outputs/qwen25-3b-toolcall/adapter
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toolft.io import load_config, read_jsonl, write_json, write_jsonl  # noqa: E402
from toolft.prompts import FEWSHOT_SYSTEM  # noqa: E402
from toolft.scoring import aggregate, score_example  # noqa: E402


def load_model(model_name: str, adapter: str | None):
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return tok, model


def build_prompt(tok, example: dict, style: str) -> str:
    messages = [{"role": "user", "content": example["messages"][0]["content"]}]
    if style == "fewshot":
        messages.insert(0, {"role": "system", "content": FEWSHOT_SYSTEM})
    return tok.apply_chat_template(messages, tools=example["tools"], add_generation_prompt=True, tokenize=False)


@torch.no_grad()
def generate_all(tok, model, prompts: list[str], batch_size: int, max_new_tokens: int) -> list[str]:
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths per batch
    outputs: list[str | None] = [None] * len(prompts)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        new_tokens = gen[:, batch["input_ids"].shape[1] :]
        texts = tok.batch_decode(new_tokens, skip_special_tokens=False)
        for i, text in zip(idx, texts):
            outputs[i] = text
        print(f"  generated {min(start + batch_size, len(order))}/{len(order)}", flush=True)
    return outputs  # type: ignore[return-value]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--name", required=True, help="run name; results go to results/<name>/")
    ap.add_argument("--adapter", help="path to a LoRA adapter directory (omit for the base model)")
    ap.add_argument("--prompt-style", choices=["native", "fewshot"], default="native")
    ap.add_argument("--data", help="eval.jsonl path (default: from config)")
    ap.add_argument("--limit", type=int, help="evaluate only the first N records (smoke test)")
    ap.add_argument("--batch-size", type=int)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_path = Path(args.data or Path(cfg["data"]["out_dir"]) / "eval.jsonl")
    examples = read_jsonl(data_path)
    if args.limit:
        examples = examples[: args.limit]

    tok, model = load_model(cfg["model"]["name"], args.adapter)
    prompts = [build_prompt(tok, ex, args.prompt_style) for ex in examples]
    outputs = generate_all(
        tok,
        model,
        prompts,
        args.batch_size or cfg["eval"]["batch_size"],
        cfg["eval"]["max_new_tokens"],
    )

    scored, preds = [], []
    for ex, text in zip(examples, outputs):
        s = score_example(ex, text)
        scored.append(s)
        preds.append({"id": ex["id"], "output": text, **s})

    metrics = aggregate(scored)
    metrics["run"] = {
        "name": args.name,
        "model": cfg["model"]["name"],
        "adapter": args.adapter,
        "prompt_style": args.prompt_style,
        "data": str(data_path),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "decoding": "greedy",
        "max_new_tokens": cfg["eval"]["max_new_tokens"],
    }

    out = Path(cfg["eval"]["results_dir"]) / args.name
    write_json(out / "metrics.json", metrics)
    write_jsonl(out / "predictions.jsonl", preds)

    def fmt(key: str) -> str:
        r = metrics[key]
        return "n/a" if r["rate"] is None else f"{100 * r['rate']:.1f}% (n={r['n']})"

    print(f"\nRun {args.name}")
    for key in ("valid_call", "tool_correct", "args_correct", "clean_format", "abstain_correct"):
        print(f"  {key:16s} {fmt(key)}")
    print(f"Wrote {out}/metrics.json and predictions.jsonl")


if __name__ == "__main__":
    main()
