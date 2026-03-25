"""Tests for the build pipeline: compile graph → transformer → inference."""

import os

import pytest
import torch

from transformer_vm.graph.core import reset_graph


@pytest.fixture(scope="module")
def built_model():
    """Build the WASM model once for all tests in this module."""
    reset_graph()
    import transformer_vm.evaluator as _evaluator_mod

    _evaluator_mod._default_graph = None

    from transformer_vm.model.weights import build_model
    from transformer_vm.wasm.interpreter import WASMMachine

    pg = WASMMachine().build()
    model, all_tokens, tok_to_idx_map, _ = build_model(program_graph=pg, plan_path=None)
    return model, all_tokens, tok_to_idx_map


def _run_and_compare(model, all_tokens, tok_to_idx_map, prog_file, ref_file, max_new_tokens=3000):
    """Run model on a program file and compare to reference."""
    with open(prog_file) as f:
        tokens = f.read().split()
    idx_seq = [tok_to_idx_map[t] for t in tokens]
    result = model.generate_with_cache(
        torch.tensor([idx_seq], dtype=torch.long), max_new_tokens=max_new_tokens
    )
    predicted = [all_tokens[i] for i in result[0].tolist()]

    with open(ref_file) as f:
        ref_tokens = f.read().split()
    return predicted, ref_tokens


@pytest.mark.slow
@pytest.mark.parametrize("program", ["hello", "collatz", "fibonacci"])
def test_build(built_model, data_dir, program):
    """Built model produces correct output for each program."""
    model, all_tokens, tok_to_idx_map = built_model
    prog = os.path.join(data_dir, f"{program}.txt")
    ref = os.path.join(data_dir, f"{program}_ref.txt")
    assert os.path.exists(prog), f"Missing {prog}"
    assert os.path.exists(ref), f"Missing {ref} — run: uv run wasm-reference"
    predicted, ref_tokens = _run_and_compare(
        model, all_tokens, tok_to_idx_map, prog, ref, max_new_tokens=50000
    )
    assert predicted == ref_tokens
