.PHONY: check clean-pycache distclean mypy test format ruff fix
SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules
PROJECT_DIRS = python_template tests

install:
	uv venv
	source .venv/bin/activate
	uv lock
	uv sync

check: ruff mypy

ruff:
	@echo "============================== Linting and Formatting check ==================="
	ruff format --check $(PROJECT_DIRS) 
	ruff check $(PROJECT_DIRS)

fix: format
	ruff check --fix $(PROJECT_DIRS)

mypy:
	@echo "============================== Type check ========================="
	mypy $(PROJECT_DIRS) 

test:
	py.test -vv

coverage:
	@echo "============================== Tests =============================="
	py.test -vv --cov --junitxml=.out/test_results.xml --cov-report "xml:.out/coverage.xml"
	@echo " => Open HTML coverage report in browser: .out/htmlcov/index.html"

format:
	ruff format $(PROJECT_DIRS)

requirements:
	uv export --no-hashes --no-dev > requirements.txt
	uv export --no-hashes > requirements-dev.txt

clean-pycache:
	find . -type d -name '__pycache__' -print0 | xargs -0 -I {} /bin/rm -rf "{}"

distclean: clean-pycache
	rm -rf dist

badges:
	make coverage
	genbadge tests -i .out/test_results.xml -o .html/badges/tests-badge.svg
	genbadge coverage -i .out/coverage.xml -o .html/badges/coverage-badge.svg
