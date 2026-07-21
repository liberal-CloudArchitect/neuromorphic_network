.PHONY: env env-update lint format-check typecheck test smoke smoke-mps gate0 smoke-p1 smoke-p2-ci smoke-p2-mps smoke-p3-ci qualify-p3-mps smoke-p4-ci qualify-p4-mps p3-full-start p3-full-status p3-full-logs p3-full-resume p3-full-stop p3-full-verify p4-start p4-status p4-logs p4-resume p4-stop p4-verify p4-record-ci check

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

gate0:
	$(CONDA_RUN) pytest tests/unit/test_contracts.py tests/integration/test_telemetry_schema.py tests/integration/test_project_assets.py

smoke-p1:
	$(CONDA_RUN) python -m neuromorphic.training.run --config configs/experiments/p1/associative_recall_smoke.yaml
	$(CONDA_RUN) python -m neuromorphic.training.run --config configs/experiments/p1/delayed_rule_switch_smoke.yaml
	$(CONDA_RUN) python -m neuromorphic.training.run --config configs/experiments/p1/small_graph_smoke.yaml

smoke-p2-ci:
	$(CONDA_RUN) python -m neuromorphic.training.run --config configs/experiments/p2/ci.yaml

smoke-p2-mps:
	$(CONDA_RUN) python -m neuromorphic.training.run --config configs/experiments/p2/gate.yaml

smoke-p3-ci:
	$(CONDA_RUN) python -m neuromorphic.training.run --config configs/experiments/p3/ci.yaml

qualify-p3-mps:
	$(CONDA_RUN) python -m neuromorphic.training.run --config configs/experiments/p3/qualification.yaml

smoke-p4-ci:
	$(CONDA_RUN) python -m neuromorphic.training.run --config configs/experiments/p4/ci.yaml

qualify-p4-mps:
	$(CONDA_RUN) python -m neuromorphic.training.run --config configs/experiments/p4/qualification.yaml

p3-full-start:
	./scripts/p3_full_run.sh start

p3-full-status:
	./scripts/p3_full_run.sh status

p3-full-logs:
	./scripts/p3_full_run.sh logs

p3-full-resume:
	./scripts/p3_full_run.sh resume

p3-full-stop:
	./scripts/p3_full_run.sh stop

p3-full-verify:
	./scripts/p3_full_run.sh verify

P4_PROFILE ?= qualification

p4-start:
	./scripts/p4_run.sh start $(P4_PROFILE)

p4-status:
	./scripts/p4_run.sh status

p4-logs:
	./scripts/p4_run.sh logs

p4-resume:
	./scripts/p4_run.sh resume

p4-stop:
	./scripts/p4_run.sh stop

p4-verify:
	./scripts/p4_run.sh verify

p4-record-ci:
	./scripts/p4_run.sh record-ci

check: lint format-check typecheck test smoke
