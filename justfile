default:
    @just --list

sync:
    uv sync --all-groups

format:
    uv run ruff format . --fix

lint:
    uv run ruff check . --fix

typecheck:
    uv run ty check . --fix

test:
    uv run pytest

check: lint typecheck test

integration:
    uv run pytest tests/integration -m 'not performance'

