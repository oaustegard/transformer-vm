#!/usr/bin/env python3
"""Build the universal WASM transformer with analytically derived weights.

Usage:
    python -m transformer_vm.build
    python -m transformer_vm.build --milp
    python -m transformer_vm.build --save-weights=model.bin
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def build(plan_path=None, max_layers=None, no_reuse=False, max_ffn=None):
    """Build the universal WASM transformer.

    Returns (model, all_tokens, tok_to_idx_map).
    """
    from transformer_vm.model.weights import build_model

    model, all_tokens, tok_to_idx_map, _ = build_model(
        plan_path=plan_path, max_layers=max_layers, no_reuse=no_reuse, max_ffn=max_ffn
    )
    return model, all_tokens, tok_to_idx_map


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Build the universal WASM transformer with analytically derived weights."
    )
    parser.add_argument(
        "--plan", type=str, default=None, help="Schedule plan YAML path (omit to run MILP solver)"
    )
    parser.add_argument(
        "--milp",
        action="store_true",
        help="Generate optimal schedule via MILP solver (ignores --plan)",
    )
    parser.add_argument(
        "--layers",
        "--max-layers",
        type=int,
        default=None,
        dest="max_layers",
        help="Max transformer layers",
    )
    parser.add_argument("--max-ffn", type=int, default=None, help="Max FFN neurons per layer")
    parser.add_argument("--no-reuse", action="store_true", help="Disable slot reuse")
    parser.add_argument(
        "--save-weights", type=str, default=None, help="Save weights to binary file"
    )
    args = parser.parse_args()

    plan = None if args.milp else args.plan
    model, all_tokens, tok_to_idx_map = build(
        plan_path=plan, max_layers=args.max_layers, no_reuse=args.no_reuse, max_ffn=args.max_ffn
    )

    d_model = model.tok.weight.shape[1]
    n_layers = len(model.attn)
    d_ffn = model.ff_in[0].weight.shape[0] // 2
    n_heads = model.attn[0].num_heads
    n_params = sum(p.numel() for p in model.parameters())

    logger.info("Universal model:")
    logger.info(
        "  d_model=%d, n_layers=%d, n_heads=%d, d_ffn=%d", d_model, n_layers, n_heads, d_ffn
    )
    logger.info("  vocab=%d, params=%s", len(all_tokens), f"{n_params:,}")

    if args.save_weights:
        from transformer_vm.model.weights import save_weights

        save_weights(model, all_tokens, args.save_weights)


if __name__ == "__main__":
    main()
