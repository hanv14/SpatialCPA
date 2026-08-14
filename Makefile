.PHONY: help install test test-all lint format typecheck check clean

PY ?= python3
PKG := spatialcpav25_gen

help:
	@echo "install    editable install with dev extras"
	@echo "test       fast suite (pytest -m 'not slow'); must stay under 3 min on CPU"
	@echo "test-all   full suite, slow tests included"
	@echo "lint       ruff check + format check"
	@echo "format     ruff format + fix"
	@echo "typecheck  mypy --strict on $(PKG)"
	@echo "check      lint + typecheck + test"

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest tests/ -m "not slow"

test-all:
	$(PY) -m pytest tests/

lint:
	$(PY) -m ruff check $(PKG) tests scripts
	$(PY) -m ruff format --check $(PKG) tests scripts

format:
	$(PY) -m ruff format $(PKG) tests scripts
	$(PY) -m ruff check --fix $(PKG) tests scripts

typecheck:
	$(PY) -m mypy --strict $(PKG)

check: lint typecheck test

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find $(PKG) tests -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
