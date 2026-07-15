.PHONY: env env-update lint format-check typecheck test smoke smoke-mps check

CONDA_ENV := brain
CONDA_RUN := conda run --no-capture-output -n $(CONDA_ENV)

env:
	conda env create --solver libmamba -f environment.yml
	$(CONDA_RUN) python -m pip install --upgrade-strategy only-if-needed --editable '.[dev]'

env-update:
	conda env update --solver libmamba -n $(CONDA_ENV) -f environment.yml --prune
	$(CONDA_RUN) python -m pip install --upgrade-strategy only-if-needed --editable '.[dev]'

lint:
	$(CONDA_RUN) ruff check .

format-check:
	$(CONDA_RUN) ruff format --check .

typecheck:
	$(CONDA_RUN) mypy src scripts tests

test:
	$(CONDA_RUN) pytest

smoke:
	$(CONDA_RUN) python scripts/check_environment.py --require cpu

smoke-mps:
	$(CONDA_RUN) python scripts/check_environment.py --require mps

check: lint format-check typecheck test smoke
