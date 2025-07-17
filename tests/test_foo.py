from src.bs_translator_backend.foo import foo


def test_foo():
    assert foo("foo") == "foo"
