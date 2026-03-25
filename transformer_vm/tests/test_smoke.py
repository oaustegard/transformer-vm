"""Smoke tests: end-to-end pipeline verification.

These tests verify the full pipeline works after the reorganization:
graph evaluator, model building, and WASM compilation.
"""

import os

import pytest

from transformer_vm.evaluator import run_program


def test_graph_evaluator_hello(data_dir):
    """Graph evaluator produces correct output for hello."""
    prog = os.path.join(data_dir, "hello.txt")
    ref = os.path.join(data_dir, "hello_ref.txt")
    assert os.path.exists(prog), f"Missing {prog}"
    assert os.path.exists(ref), f"Missing {ref} — run: uv run wasm-reference"
    assert run_program(prog, ref, use_hull=True) is True


def test_graph_evaluator_collatz(data_dir):
    """Graph evaluator produces correct output for collatz."""
    prog = os.path.join(data_dir, "collatz.txt")
    ref = os.path.join(data_dir, "collatz_ref.txt")
    assert os.path.exists(prog), f"Missing {prog}"
    assert os.path.exists(ref), f"Missing {ref} — run: uv run wasm-reference"
    assert run_program(prog, ref, use_hull=True) is True


@pytest.mark.slow
def test_graph_evaluator_fibonacci(data_dir):
    """Graph evaluator produces correct output for fibonacci."""
    prog = os.path.join(data_dir, "fibonacci.txt")
    ref = os.path.join(data_dir, "fibonacci_ref.txt")
    assert os.path.exists(prog), f"Missing {prog}"
    assert os.path.exists(ref), f"Missing {ref} — run: uv run wasm-reference"
    assert run_program(prog, ref, use_hull=True) is True


def test_wasm_machine_builds():
    """WASMMachine.build() produces a valid ProgramGraph."""
    from transformer_vm.wasm.interpreter import WASMMachine

    pg = WASMMachine().build()
    assert pg.input_tokens is not None
    assert pg.output_tokens is not None
    assert len(pg.all_dims) > 0
    assert len(pg.all_lookups) > 0


def test_graph_imports():
    """All key graph primitives are importable from the new package."""
    from transformer_vm.graph.core import (
        Expression,
        InputDimension,
        one,
    )

    # Verify basic Expression arithmetic
    x = InputDimension("x")
    expr = x * 3 + one * 2
    assert isinstance(expr, Expression)
    assert expr[x] == 3
    assert expr[one] == 2


@pytest.mark.slow
def test_build_model(data_dir):
    """build_model() produces a transformer with expected shape."""
    from transformer_vm.model.weights import build_model

    model, all_tokens, tok_to_idx, _ = build_model(plan_path=None)

    # Verify model structure
    n_layers = len(model.attn)
    assert n_layers > 0
    assert len(all_tokens) > 0
    assert len(tok_to_idx) == len(all_tokens)


@pytest.mark.slow
@pytest.mark.integration
def test_model_inference_hello(data_dir):
    """Build model and run inference on hello, comparing to reference."""
    import torch

    from transformer_vm.model.weights import build_model

    model, all_tokens, tok_to_idx, _ = build_model(plan_path=None)

    prog_file = os.path.join(data_dir, "hello.txt")
    ref_file = os.path.join(data_dir, "hello_ref.txt")
    assert os.path.exists(prog_file), f"Missing {prog_file}"
    assert os.path.exists(ref_file), f"Missing {ref_file} — run: uv run wasm-reference"

    with open(prog_file) as f:
        prog_tokens = f.read().split()

    idx_list = [tok_to_idx[t] for t in prog_tokens]
    idx = torch.tensor([idx_list], dtype=torch.long)

    out = model.generate_with_cache(idx, max_new_tokens=5000)
    out_tokens = [all_tokens[i] for i in out[0].tolist()]

    with open(ref_file) as f:
        ref_tokens = f.read().split()

    assert out_tokens == ref_tokens
