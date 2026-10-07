from importlib import import_module


def test_package_is_importable():
    package = import_module("ragforge")
    assert package.__name__ == "ragforge"
