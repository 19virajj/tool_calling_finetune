ADAPTER ?= outputs/qwen25-3b-toolcall/adapter

.PHONY: data train eval-base eval-fewshot eval-tuned eval report test smoke

data:
	python scripts/prepare_data.py

train:
	python scripts/train.py

eval-base:
	python scripts/evaluate.py --name base

eval-fewshot:
	python scripts/evaluate.py --name base_fewshot --prompt-style fewshot

eval-tuned:
	python scripts/evaluate.py --name tuned --adapter $(ADAPTER)

eval: eval-base eval-fewshot eval-tuned

report:
	python scripts/make_report.py

test:
	python -m pytest -q

# Quick end-to-end check before a full run: 30 training steps, 40 eval records.
smoke:
	python scripts/train.py --max-steps 30 --output-dir outputs/smoke
	python scripts/evaluate.py --name smoke_tuned --adapter outputs/smoke/adapter --limit 40
