# Tool-calling fine-tune: Qwen2.5 + QLoRA (Unsloth, TRL)

Fine-tune a small open model so that its tool calls are well-formed and correct, and measure the
improvement honestly: on tools the model never saw in training, against both a raw baseline and a
well-prompted baseline, with confidence intervals.

## Goal (Multiple iterations)

| | Valid-call rate on held-out tools |
|---|---|
| Baseline (current) | about 71% |
| Fine-tuned (target) | about 90% |

These two numbers are the goal I set before running anything. They live in
`configs/default.yaml` under `targets`, and `make report` prints measured results next to them and
says whether the target was met. Measured results go in `results/REPORT.md`.

**Status: not run yet.** The results section below stays empty until `make eval` and `make report` have run.

## How it works

```
prepare_data.py   xlam-function-calling-60k -> train.jsonl / eval.jsonl   (held-out tools, abstain slice)
train.py          QLoRA on Qwen2.5-3B-Instruct, loss on the tool call only
evaluate.py       generate with greedy decoding, score every output     (base, base+few-shot, tuned)
make_report.py    results/REPORT.md + results/comparison.png
```

Everything shares one validator (`src/toolft/schema.py`), so the gold labels, the training data
and the model outputs are all judged by the same rules.

### Metrics (call slice)

| Metric | Meaning |
|---|---|
| Valid call | Output parses, has at least one call, and every call names a known tool, includes all required arguments, has no unknown arguments, and uses the right JSON types. A structural check. |
| Right tool | The multiset of tool names equals the gold calls. |
| Right args | Right tool and the argument values equal the gold values exactly (order-insensitive, `1.0 == 1`). The strict metric. |
| Clean format | Nothing but `<tool_call>` blocks in the output (no surrounding prose). |

On the abstain slice (no provided tool fits the request), **abstain correct** means the model emitted no tool call.

Report valid call together with right args. Valid call alone only shows that the structure is right.

### Design decisions that keep the number honest

- **Held-out tools, not just held-out rows.** Tool names are hashed into train and test buckets. A training example never contains a test-bucket tool anywhere in its tool list, and every eval example's gold calls use only test-bucket tools. Examples that mix the two are dropped. The eval therefore measures generalisation to unseen tools.
- **Gold labels are validated.** Rows whose gold answer fails the validator are dropped, so a perfect score is reachable and the validator is calibrated to the data.
- **Two baselines.** The base model is evaluated zero-shot and with a few-shot system prompt (`src/toolft/prompts.py`). The fine-tune has to beat good prompting, not only the raw model.
- **Abstain slice.** The query is paired with another example's tool list that lacks the right tool. Training includes a small share of these so the model does not learn to always emit a call, and the eval checks it did not regress.
- **Same prompt format everywhere.** Training and evaluation both render prompts with the model's own chat template, including the tool list.
- **Confidence intervals.** 95% Wilson intervals on every rate.

## Setup

Needs an NVIDIA GPU for `train.py` and `evaluate.py`. A Colab T4 is enough for the 3B model with 4-bit QLoRA.

```bash
git clone https://github.com/19virag/tool_calling_finetune
cd tool_calling_finetune
pip install -r requirements.txt     # on Colab: pip install unsloth first, torch is preinstalled
```

The dataset is gated on Hugging Face. Open the dataset page, accept the terms, then log in:

```bash
huggingface-cli login     # or: hf auth login
```

Data prep, scoring and the report need no GPU. To run only the tests: `pip install pyyaml matplotlib pytest && make test`.

## Run

```bash
make test             # unit tests, no GPU
make data             # build data/processed/{train,eval}.jsonl and stats.json
make smoke            # 30 training steps + 40 eval records: checks the whole GPU path first
make train            # full QLoRA run -> outputs/qwen25-3b-toolcall/adapter
make eval             # base, base_fewshot, tuned -> results/<name>/
make report           # results/REPORT.md and results/comparison.png
```

Run `make smoke` before the full run. The training and generation scripts depend on Unsloth, TRL
and transformers versions that change quickly, so the first run on a new machine is where
version mismatches show up. `train.py` already handles the `max_seq_length` / `max_length` and
`tokenizer` / `processing_class` renames in TRL.

Everything is configured in `configs/default.yaml` (model, LoRA rank, training size, eval size, test-tool share).

## Layout

```
configs/default.yaml      all settings and the written-down targets
scripts/prepare_data.py   dataset -> train/eval JSONL, held-out tool split, abstain slice
scripts/train.py          Unsloth + TRL QLoRA training
scripts/evaluate.py       generation + scoring for any model/adapter/prompt style
scripts/make_report.py    report and chart from results/*/metrics.json
src/toolft/               schema normalisation + validation, output parsing, scoring, few-shot prompt
tests/                    unit tests on synthetic data (no network, no GPU)
results/                  metrics.json and predictions.jsonl per run, REPORT.md, comparison.png
```

## Limitations

- Single-turn tool calls only. No multi-turn conversations, tool responses, or tool use inside longer chats.
- "Required" versus "optional" parameters are inferred from the dataset (a parameter with a default, or an `Optional` type, is optional). The same rule is applied to gold labels, training data and scoring, but it is a heuristic.
- The abstain slice is synthetic: a mismatched tool list is assumed not to contain a usable tool. Occasionally a swapped-in tool may still be relevant.
- Training uses a 4-bit base (QLoRA); evaluation merges the adapter into a bf16 base. This is the usual setup, but it is a small train/inference mismatch.
- Valid call is structural. It says nothing about whether the argument values are right; use Right args for that.
- Results come from one run with one seed and greedy decoding. Run more seeds before claiming small differences.

## Results

Not run yet. After `make eval && make report`, paste the table from `results/REPORT.md` here and
link `results/comparison.png`. Report the measured numbers even if they differ from the targets above.
