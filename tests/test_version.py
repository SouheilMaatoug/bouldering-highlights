import importlib

from src import bouldering


def test_package_version():
    assert bouldering.__version__ == importlib.metadata.version("bouldering")
