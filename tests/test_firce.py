def test_import_core():
    try:
        import firce  # noqa: F401
    except ImportError:
        assert False, 'Failed to import core'
    assert True
