# 🧪 tests/

This directory contains unit tests for verifying the functionality and import integrity of the `core` and `FIRE` modules.

---

## ✅ Test Structure

* `test_core.py`: Verifies that the core runtime components (e.g., `listener.py`) can be imported successfully.
* `test_FIRE.py`: Checks that the FIRE ML pipeline modules (`main.py`, `models.py`, etc.) can be imported without errors.
* `__init__.py`: Marks this directory as a Python package.

---

## 🚀 Running Tests

To run all tests:

```bash
pytest
```

Ensure you're in the project root and have `pytest` installed via:

```bash
uv pip install pytest
```

Project uses a `pytest.ini` file to set the `src/` directory as the import root, so tests can directly access internal modules like:

```python
from FIRE import main
from core import listener
```

---

## 🛠️ Guidelines

* Place all new test files in this directory.
* Use descriptive names: `test_<module>.py`
* Prefer `assert`-style testing and include meaningful test names.

---