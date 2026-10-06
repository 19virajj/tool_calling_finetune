#!/usr/bin/env python
"""QLoRA fine-tune of a Qwen instruct model for tool calling, using Unsloth + TRL.

Requires an NVIDIA GPU. Loss is computed on the assistant turn only.
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

# Unsloth must be imported before transformers / trl so its patches apply.
from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: E402
from unsloth.chat_templates import train_on_responses_only  # noqa: E402

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from toolft.io import load_config, read_jsonl, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-steps", type=int, help="override train.max_steps (smoke test)")
    ap.add_argument("--output-dir", help="override train.output_dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    m, lo, t = cfg["model"], cfg["lora"], cfg["train"]
    seed = cfg["seed"]
    out_dir = Path(args.output_dir or t["output_dir"])
    max_steps = args.max_steps if args.max_steps is not None else t["max_steps"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m["name"],
        max_seq_length=m["max_seq_length"],
        dtype=None,  # auto: bf16 where supported, else fp16
        load_in_4bit=m["load_in_4bit"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lo["r"],
        target_modules=lo["target_modules"],
        lora_alpha=lo["alpha"],
        lora_dropout=lo["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    # Render each record with the model's own chat template, tools included, so training
    # and evaluation see exactly the same prompt format.
    records = read_jsonl(Path(cfg["data"]["out_dir"]) / "train.jsonl")
    texts = [
        tokenizer.apply_chat_template(r["messages"], tools=r["tools"], tokenize=False, add_generation_prompt=False)
        for r in records
    ]
    # Build the dataset from plain strings: nested tool arguments have heterogeneous shapes,
    # which Arrow cannot always unify.
    dataset = Dataset.from_dict({"text": texts})
    print(f"{len(dataset)} training examples. Sample:\n{texts[0][-600:]}\n")

    use_bf16 = is_bfloat16_supported()
    sft_kwargs = dict(
        output_dir=str(out_dir),
        dataset_text_field="text",
        per_device_train_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["grad_accum_steps"],
        num_train_epochs=t["epochs"],
        max_steps=max_steps,
        learning_rate=t["learning_rate"],
        warmup_steps=t["warmup_steps"],
        weight_decay=t["weight_decay"],
        lr_scheduler_type=t["lr_scheduler"],
        optim=t["optim"],
        logging_steps=t["logging_steps"],
        bf16=use_bf16,
        fp16=not use_bf16,
        seed=seed,
        save_strategy="no",
        report_to="none",
        packing=False,
    )
    # TRL renamed max_seq_length -> max_length in newer releases.
    length_key = "max_seq_length" if "max_seq_length" in inspect.signature(SFTConfig).parameters else "max_length"
    sft_kwargs[length_key] = m["max_seq_length"]
    sft_config = SFTConfig(**sft_kwargs)

    try:
        trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=dataset, args=sft_config)
    except TypeError:  # older TRL
        trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=sft_config)

    # Only learn from the assistant turn (the tool call), not the system prompt / tool list / user turn.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    stats = trainer.train()

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    write_json(
        out_dir / "train_summary.json",
        {
            "config": cfg,
            "n_examples": len(dataset),
            "train_runtime_s": stats.metrics.get("train_runtime"),
            "final_train_loss": stats.metrics.get("train_loss"),
            "log_history": trainer.state.log_history,
            "torch": torch.__version__,
        },
    )
    print(f"Adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
