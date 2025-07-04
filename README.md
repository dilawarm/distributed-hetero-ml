# Python Template

A basic Python project template that actually works. Nothing fancy, just the stuff you need to get going.

## Table of Contents

- [What's in here](#whats-in-here)
- [Getting started](#getting-started)
- [Development](#development)
- [Docker](#docker)
- [Testing](#testing)
- [Code quality](#code-quality)
- [GitHub Actions](#github-actions)
- [Project structure](#project-structure)

## What's in here

This template gives you a Python 3.13 project with:

- **uv** for dependency management (it's fast)
- **pytest** for testing
- **ruff** for linting and formatting
- **mypy** for type checking
- **Docker** setup that actually builds
- **GitHub Actions** for CI/CD
- **Makefile** for common tasks

## Getting started

Clone this template and make it your own:

```bash
git clone <your-repo>
cd <your-project>
```

or just click the `Use this template` button.

Install dependencies:

```bash
uv run make install
```

That's it. You're ready to code.

## Development

The main code goes in `python_template/`. There's already a simple function in `main.py` to get you started.

### Common commands

```bash
# Install everything
uv run make install

# Run tests
uv run make test

# Check your code (linting + type checking)
uv run make check

# Fix formatting issues
uv run make fix

# Run tests with coverage
uv run make coverage
```

### Adding dependencies

Add them to `pyproject.toml` and run:

```bash
uv lock
uv sync
```

## Docker

Build and run with Docker:

```bash
# Build the image
docker build -t python-template .

# Run it
docker run -it python-template /bin/bash

# Run tests in Docker
docker run --rm python-template uv run make test
```

## Testing

Tests go in the `tests/` folder. The template includes a basic test to make sure everything works.

Run tests:

```bash
uv run make test
```

Get coverage:

```bash
uv run make coverage
```

## Code quality

The template enforces code quality with:

- **ruff** for linting and formatting
- **mypy** for type checking
- **pre-commit** hooks (optional but recommended)

Check everything:

```bash
uv run make check
```

Fix formatting:

```bash
uv run make fix
```

## GitHub Actions

Three workflows are set up:

- **CI**: Runs on every push/PR. Tests your code, checks formatting, runs security scans.
- **Docker**: Builds and publishes Docker images to GitHub Container Registry.
- **Release**: Creates releases when you push version tags.

Just push your code and everything runs automatically.

### Making a release

```bash
git tag v1.0.0
git push origin v1.0.0
```

This triggers the release workflow and publishes your Docker image.

## Project structure

```
python-template/
├── python_template/          # Your main code
│   ├── __init__.py
│   └── main.py
├── tests/                    # Your tests
│   ├── __init__.py
│   └── test_main.py
├── .github/workflows/        # GitHub Actions
├── Dockerfile               # Docker setup
├── Makefile                 # Common commands
├── pyproject.toml           # Project config and dependencies
├── requirements.txt         # Frozen dependencies
└── README.md               # This file
```

---

That's pretty much it. This template handles the boring setup stuff so you can focus on writing code. Change the name in `pyproject.toml`, update this README, and start building. 