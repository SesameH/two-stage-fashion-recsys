.PHONY: setup download data baselines recall train analyze rolling serve-data serve images bench test lint clean
export PYTHONPATH := .

PY := .venv/bin/python
PIP := .venv/bin/pip

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

download:
	@test -f ~/.kaggle/kaggle.json || \
		(echo "Missing ~/.kaggle/kaggle.json. Create an API token at kaggle.com/settings" && exit 1)
	.venv/bin/kaggle competitions download \
		-c h-and-m-personalized-fashion-recommendations \
		-f transactions_train.csv -p data/raw
	.venv/bin/kaggle competitions download \
		-c h-and-m-personalized-fashion-recommendations \
		-f customers.csv -p data/raw
	.venv/bin/kaggle competitions download \
		-c h-and-m-personalized-fashion-recommendations \
		-f articles.csv -p data/raw
	cd data/raw && for f in *.zip; do [ -e "$$f" ] && unzip -o "$$f" && rm "$$f"; done; true

data:
	$(PY) -m src.data.ingest

baselines:
	$(PY) -m src.baselines.run

recall:
	$(PY) -m src.recall.run

train:
	$(PY) -m src.rank.train

analyze:
	$(PY) -m src.eval.analyze

rolling:
	$(PY) -m src.eval.rolling

serve-data:
	$(PY) -m src.serve.precompute --with-demo

serve:
	@echo "console at http://localhost:8000/"
	.venv/bin/uvicorn src.serve.app:app --host 0.0.0.0 --port 8000

# Local demo / README screenshots only. Output is excluded from git and from the
# Docker image: these are H&M's product photographs, not ours to redistribute.
images:
	$(PY) -m src.serve.fetch_images

bench:
	$(PY) -m src.serve.bench --n 400

lint:
	.venv/bin/ruff check src tests

test:
	.venv/bin/pytest -q

clean:
	rm -rf data/processed/* models/* reports/*
