#!/usr/bin/env python3
"""make_synthetic_model.py — Emit a synthetic model.bin in transformer.cpp's
binary format with controllable sparsity, plus a synthetic token program.

The output is functionally meaningless (random weights), but the format is
valid and the per-token forward pass exercises identical code paths to a
real model.bin, which is exactly what we need to benchmark dense-projection
vs sparse-projection.
"""
from __future__ import annotations

import argparse
import struct
import sys

import numpy as np


def sparse_random(rows: int, cols: int, sparsity: float, rng: np.random.Generator) -> np.ndarray:
    """rows×cols matrix with `sparsity` fraction of entries set to 0."""
    m = rng.standard_normal((rows, cols)) * 0.01
    if sparsity > 0:
        mask = rng.random((rows, cols)) < sparsity
        m[mask] = 0.0
    return m.astype(np.float64)


def build_vocab(V: int) -> list[str]:
    """A vocabulary with enough structure to exercise the engine.

    Reserves a stop token at index V-1 so the engine has somewhere to halt.
    """
    names = []
    # Reserve common opcodes so the runner.py doesn't choke
    reserved = ["{", "}", " ", "halt", "branch_taken", "call_commit", "return_commit"]
    names.extend(reserved)
    # Fill the rest with synthetic tokens
    for i in range(V - len(reserved) - 1):
        names.append(f"t{i:05d}")
    names.append("STOP")  # last token is stop
    assert len(names) == V
    return names


def write_model(
    path: str,
    V: int,
    D: int,
    L: int,
    H: int,
    F: int,
    sparsity: float,
    seed: int = 42,
) -> None:
    """Emit binary in transformer.cpp::load() format."""
    rng = np.random.default_rng(seed)
    names = build_vocab(V)
    stop_idx = V - 1

    with open(path, "wb") as f:
        # Header: V D L H F stop  (6 × int32)
        f.write(struct.pack("<6i", V, D, L, H, F, stop_idx))

        # Vocabulary: V × (uint32 length + bytes)
        for name in names:
            b = name.encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)

        # Weights:
        #   embedding:        V × D
        #   per layer L of:
        #       qkv:          3D × D
        #       out:          D × D
        #       fi:           2F × D
        #       fo:           D × F
        #   head:             V × D
        emb = sparse_random(V, D, sparsity, rng)
        f.write(emb.tobytes())

        for _ in range(L):
            qkv = sparse_random(3 * D, D, sparsity, rng)
            out = sparse_random(D, D, sparsity, rng)
            fi = sparse_random(2 * F, D, sparsity, rng)
            fo = sparse_random(D, F, sparsity, rng)
            for w in (qkv, out, fi, fo):
                f.write(w.tobytes())

        head = sparse_random(V, D, sparsity, rng)
        f.write(head.tobytes())

        # No erase lists, no tie-break overrides
        f.write(struct.pack("<i", 0))  # has_erase = false
        f.write(struct.pack("<i", 0))  # has_tb = false

    # Report what we wrote
    nonzero_per_proj = int((1 - sparsity) * (3 * D * D + D * D + 2 * F * D + D * F))
    print(
        f"Wrote {path}: V={V} D={D} L={L} H={H} F={F} "
        f"sparsity={sparsity:.2f} "
        f"per-layer-nonzero≈{nonzero_per_proj:,}"
    )


def write_program(path: str, model_bin: str, n_input_tokens: int, n_bench_steps: int, seed: int = 1) -> None:
    """Write a token program the engine will read, plus a stub *_ref.txt
    that bounds the engine's MAX_GEN so the run is deterministic and finite.

    The engine computes max_gen = len(ref) - plen + 100, so writing a ref
    of length n_bench_steps gives roughly n_bench_steps generated tokens.
    Ref content is irrelevant (the engine warns on mismatch but doesn't exit
    in regen-less mode; we run with regen to suppress the diff entirely).
    """
    rng = np.random.default_rng(seed)
    # Read vocab from model.bin
    with open(model_bin, "rb") as f:
        V, D, L, H, F, stop = struct.unpack("<6i", f.read(24))
        names = []
        for _ in range(V):
            (n,) = struct.unpack("<I", f.read(4))
            names.append(f.read(n).decode("utf-8"))

    # Pick "filler" tokens — t* tokens by name
    pool = [n for n in names if n.startswith("t")]
    chosen = rng.choice(pool, size=n_input_tokens, replace=True)

    with open(path, "w") as f:
        f.write("{\n")
        f.write(" ".join(chosen) + "\n")
        f.write("}\n")
    print(f"Wrote {path}: {n_input_tokens + 2} input tokens")

    # Stub ref of length n_bench_steps to bound the engine's run
    ref_path = path.replace(".txt", "_ref.txt")
    with open(ref_path, "w") as f:
        ref_tokens = rng.choice(pool, size=n_bench_steps, replace=True)
        f.write(" ".join(ref_tokens) + "\n")
    print(f"Wrote {ref_path}: {n_bench_steps} ref tokens (bounds max_gen)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth_model.bin")
    ap.add_argument("--prog", default="data/synth_prog.txt")
    ap.add_argument("--V", type=int, default=256, help="Vocab size")
    ap.add_argument("--D", type=int, default=128, help="d_model")
    ap.add_argument("--L", type=int, default=4, help="layers")
    ap.add_argument("--H", type=int, default=4, help="attention heads (each 2D)")
    ap.add_argument("--F", type=int, default=256, help="d_ffn")
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--prog-len", type=int, default=20)
    ap.add_argument("--bench-steps", type=int, default=500,
                    help="Stub _ref length; bounds engine max_gen for deterministic runs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    write_model(args.out, args.V, args.D, args.L, args.H, args.F, args.sparsity, args.seed)
    write_program(args.prog, args.out, args.prog_len, args.bench_steps, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
