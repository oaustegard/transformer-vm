#!/usr/bin/env python3
"""First Futamura Projection: specialize the WASM interpreter for a specific program.

Bakes the program's bytecode into FFN weights, eliminating the program prefix
from the input. The resulting transformer takes only execution-phase tokens
as input (start + input bytes), producing the same output as the universal
interpreter.

Usage:
    python -m transformer_vm.specialize data/collatz.txt
    python -m transformer_vm.specialize data/collatz.txt --save-weights=collatz.bin
"""

import argparse
import logging
import os

logger = logging.getLogger(__name__)


def parse_program(filepath):
    """Parse program instructions from a .txt file.

    Accepts either:
      - A full .txt file with { ... } delimiters (extracts the program between them)
      - A bare instruction file (no delimiters, every line is an instruction)

    Each instruction line is: opcode hex0 hex1 hex2 hex3

    Returns list of dicts with 'opcode' and 'bytes' keys.
    """
    with open(filepath) as f:
        content = f.read()

    lines = content.split("\n")

    # Check if this is a full .txt file with { } delimiters
    tokens = content.split()
    if "{" in tokens:
        start = tokens.index("{")
        try:
            end = tokens.index("}")
        except ValueError:
            raise ValueError(f"Found '{{' but no '}}' in {filepath}") from None
        # Extract instruction tokens between { and }
        prog_tokens = tokens[start + 1 : end]
        # Group into 5-token instructions (opcode + 4 hex bytes)
        instructions = []
        for i in range(0, len(prog_tokens), 5):
            chunk = prog_tokens[i : i + 5]
            if len(chunk) < 5:
                raise ValueError(f"Incomplete instruction at token {i}: {chunk}")
            opcode = chunk[0]
            bytes_ = [int(chunk[1 + j], 16) for j in range(4)]
            instructions.append({"opcode": opcode, "bytes": bytes_})
        return instructions

    # Bare format: each line is an instruction
    instructions = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        opcode = parts[0]
        if len(parts) < 5:
            raise ValueError(f"Incomplete instruction: {line.strip()}")
        bytes_ = [int(parts[1 + j], 16) for j in range(4)]
        instructions.append({"opcode": opcode, "bytes": bytes_})
    return instructions


def spec_input_from_txt(filepath):
    """Extract the specialized model input from a .txt file.

    Returns ['start'] + input tokens after '}' in the file.
    """
    with open(filepath) as f:
        tokens = f.read().split()
    try:
        end = tokens.index("}")
    except ValueError:
        raise ValueError(f"No '}}' found in {filepath}") from None
    return ["start"] + tokens[end + 1 :]


def specialize(filepath):
    """Specialize the WASM interpreter for the program in filepath.

    Returns (model, all_tokens, tok_to_idx_map, instructions).
    """
    instructions = parse_program(filepath)
    logger.info("Parsed %d instructions from %s", len(instructions), filepath)

    from transformer_vm.wasm.interpreter import WASMMachine

    pg = WASMMachine(program=instructions).build()
    logger.info(
        "  %d dims, %d lookups, %d input tokens, %d output tokens",
        len(pg.all_dims),
        len(pg.all_lookups),
        len(pg.input_tokens),
        len(pg.output_tokens),
    )

    from transformer_vm.model.weights import build_model

    model, all_tokens, tok_to_idx_map, _ = build_model(program_graph=pg)

    return model, all_tokens, tok_to_idx_map, instructions


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Specialize the WASM interpreter for a specific program."
    )
    parser.add_argument("program", help="Path to .txt program file")
    parser.add_argument(
        "--save-weights", type=str, default=None, help="Save specialized weights to binary file"
    )
    args = parser.parse_args()

    prog_file = args.program
    name = os.path.basename(prog_file).replace(".txt", "")

    model, all_tokens, tok_to_idx_map, instructions = specialize(prog_file)

    d_model = model.tok.weight.shape[1]
    n_layers = len(model.attn)
    d_ffn = model.ff_in[0].weight.shape[0] // 2
    n_heads = model.attn[0].num_heads
    n_params = sum(p.numel() for p in model.parameters())

    logger.info("Specialized model for %s:", name)
    logger.info(
        "  d_model=%d, n_layers=%d, n_heads=%d, d_ffn=%d", d_model, n_layers, n_heads, d_ffn
    )
    logger.info("  vocab=%d, params=%s", len(all_tokens), f"{n_params:,}")
    logger.info("  instructions baked: %d", len(instructions))

    if args.save_weights:
        from transformer_vm.model.weights import save_weights

        save_weights(model, all_tokens, args.save_weights)


if __name__ == "__main__":
    main()
