.PHONY: test lint fmt clean install dev api

test:
	pytest tests/ -v

coverage:
	pytest tests/ --cov=engine --cov=components --cov=sizing --cov=governance --cov=evaluation --cov=bridges --cov=storage --cov=config --cov=api --cov-report=term-missing

lint:
	ruff check .

fmt:
	ruff format .
	ruff check --fix .

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

api:
	uvicorn api.main:app --reload --port 8000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist build .pytest_cache htmlcov .coverage .ruff_cache *.egg-info
