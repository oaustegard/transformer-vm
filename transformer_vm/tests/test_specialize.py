"""Tests for the specialize pipeline (First Futamura Projection)."""

import os

import pytest
import torch

from transformer_vm.graph.core import reset_graph


@pytest.fixture(scope="module", params=["hello"])
def specialized_model(request, data_dir):
    """Specialize the WASM interpreter for a given program.

    Only hello is tested here — collatz/fibonacci specialization produces MILP
    problems too large to solve within CI time limits.  The universal model
    already covers those programs in test_distill.py.
    """
    program = request.param
    reset_graph()
    import transformer_vm.evaluator as _evaluator_mod

    _evaluator_mod._default_graph = None

    from transformer_vm.specialize import specialize

    prog_file = os.path.join(data_dir, f"{program}.txt")
    model, all_tokens, tok_to_idx_map, instructions = specialize(prog_file)
    return program, model, all_tokens, tok_to_idx_map


def _run_specialized_and_compare(
    model, all_tokens, tok_to_idx_map, spec_file, ref_file, max_new_tokens=50000
):
    """Run specialized model using a _spec.txt input and compare to _ref.txt."""
    with open(spec_file) as f:
        input_tokens = f.read().split()
    idx_seq = [tok_to_idx_map[t] for t in input_tokens]
    result = model.generate_with_cache(
        torch.tensor([idx_seq], dtype=torch.long), max_new_tokens=max_new_tokens
    )
    predicted = [all_tokens[i] for i in result[0].tolist()]

    with open(ref_file) as f:
        ref_tokens = f.read().split()

    return predicted, ref_tokens


def _strip_prefix(tokens):
    """Strip the program prefix ({ ... }) from a universal model token list."""
    try:
        end = tokens.index("}")
        return tokens[end + 1 :]
    except ValueError:
        return tokens


@pytest.mark.slow
def test_specialize(specialized_model, data_dir):
    """Specialized model produces correct output."""
    program, model, all_tokens, tok_to_idx_map = specialized_model
    spec_file = os.path.join(data_dir, f"{program}_spec.txt")
    ref_file = os.path.join(data_dir, f"{program}_ref.txt")
    assert os.path.exists(spec_file), f"Missing {spec_file}"
    assert os.path.exists(ref_file), f"Missing {ref_file} — run: uv run wasm-reference"
    predicted, ref_tokens = _run_specialized_and_compare(
        model, all_tokens, tok_to_idx_map, spec_file, ref_file
    )
    pred_exec = predicted[1:] if predicted and predicted[0] == "start" else predicted
    ref_exec = _strip_prefix(ref_tokens)
    assert pred_exec == ref_exec
