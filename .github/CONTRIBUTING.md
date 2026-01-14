# Contributing to FIRCE / FIRCE-MC

Thanks for contributing! This repo contains research + engineering code for streaming intrusion detection with Conformal Evaluation, adaptive chunking, and multiclass support.

## Quick start (local dev)

### Prereqs
- Python (see `.python-version` if present)
- [`uv`](https://github.com/astral-sh/uv) installed (recommended)

### Setup
```bash
uv sync
```

### Common commands

If you prefer `Makefile`, use it as the canonical interface:

```bash
make help
make lint
make test
```

You can also use Ruff/pytest directly:

```bash
uv run ruff check .
uv run ruff format .
uv run pytest -q
```

## Branching

* Create branches from `FIRCE-MC`
* Suggested naming:

  * `feat/<short-description>`
  * `fix/<short-description>`
  * `chore/<short-description>`
  * `docs/<short-description>`
> Or use whatever you like, just make sure it makes sense 

## Code style

* Follow **PEP 8** and repo lint rules
* Use **type hints** where practical
* Prefer small, testable functions
* Keep dataset/feature schema logic centralized (avoid duplicating column lists across modules)

## Testing

* Add tests for new behavior (unit tests preferred)
* Mark expensive training tests with:

  * `@pytest.mark.slow`
* Ensure tests are deterministic (set seeds when relevant)

## Data & artifacts policy

To keep the repo clean:

* **Do not commit large datasets** (use `datasets/` locally; keep it ignored)
* **Do not commit large trained artifacts** unless explicitly intended (models, checkpoints, pickles)
* If artifacts are required for CI, keep them tiny and document why.

## Pull requests

Before opening a PR:

* Run `make lint` and `make test`
* Update docs/comments where behavior changes
* Add/adjust tests for new functionality

### PR expectations

* Clear description (what/why/how)
* Links to issue(s) when applicable
* Evidence (logs, screenshots, metrics) when changing training/simulation behavior

## Reporting issues

Use the issue templates in `.github/ISSUE_TEMPLATE/` and include:

* OS + Python version
* command used
* stack trace / logs
* expected vs actual behavior
