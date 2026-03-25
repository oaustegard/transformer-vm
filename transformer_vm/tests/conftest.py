import pytest

import transformer_vm.evaluator as _evaluator_mod
from transformer_vm._paths import DATA_DIR, EXAMPLES_DIR
from transformer_vm.graph.core import reset_graph


@pytest.fixture(autouse=True)
def clean_graph():
    """Reset graph state before and after each test."""
    reset_graph()
    _evaluator_mod._default_graph = None
    yield
    reset_graph()
    _evaluator_mod._default_graph = None


@pytest.fixture(scope="session", autouse=True)
def ensure_compiled():
    """Ensure data and reference files exist before tests run."""
    from transformer_vm.compilation.compile_wasm import ensure_data

    ensure_data()
    from transformer_vm.wasm.reference import generate_all

    generate_all()


@pytest.fixture(scope="session")
def data_dir():
    """Path to the data/ directory."""
    return DATA_DIR


@pytest.fixture(scope="session")
def examples_dir():
    """Path to the examples/ directory."""
    return EXAMPLES_DIR
