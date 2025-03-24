def test_import_core():
    try:
        import src.core  # noqa: F401
    except ImportError:
        assert False, "Failed to import core"
    assert True
