PYTHON := uv run python
SRC_DIR := src

.PHONY: all install run debug clean lint lint-strict test

all: install

install:
		uv sync

run:
		$(PYTHON) -m $(SRC_DIR)

clean:
		rm -rf .pytest_cache .mypy_cache .ruff_cache
		find . -type d -name "__pycache__" -exec rm -rf {} +
		find . -type f -name "*.pyc" -delete

lint:
		uv run flake8 $(SRC_DIR)
		uv run mypy --strict $(SRC_DIR)

test:
		uv run pytest tests/
