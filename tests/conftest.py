"""Make `left` importable from a plain checkout.

`pip install -e .` is the supported path (pyproject maps `left` -> `src/`), but
tests should also run in a bare clone, so if the package is not installed we
register `src/` under the name `left` here.
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if importlib.util.find_spec("left") is None:
    spec = importlib.util.spec_from_file_location(
        "left", os.path.join(ROOT, "src", "__init__.py"),
        submodule_search_locations=[os.path.join(ROOT, "src")])
    module = importlib.util.module_from_spec(spec)
    sys.modules["left"] = module
    spec.loader.exec_module(module)

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def repo_root():
    return ROOT


@pytest.fixture(scope="session")
def artifact_path(repo_root):
    """The shipped coefficient artifact -- the paper's transferred curve."""
    return os.path.join(repo_root, "artifacts", "fact", "imagenet1k_vitd6_seed1.json")
