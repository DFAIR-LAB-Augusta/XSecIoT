def test_import_FIRE():
    try:
        from src.FIRE import main, models, preprocessing  # noqa: F401
    except ImportError:
        assert False, "Failed to import FIRE modules"
    assert True