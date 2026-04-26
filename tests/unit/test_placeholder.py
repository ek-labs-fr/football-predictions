def test_project_imports() -> None:
    import src.api
    import src.data
    import src.features
    import src.models

    assert src.data is not None
    assert src.features is not None
    assert src.models is not None
    assert src.api is not None
