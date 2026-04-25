#!/usr/bin/env python3
"""measure_sparsity.py — Read a model.bin produced by ``save_weights()`` and
report nonzero fractions per projection per layer.

The format mirrors ``transformer_vm/model/transformer.cpp::load()``:

    header: V D L H F stop  (6 × int32, little-endian)
    vocab:  V × (uint32 length + UTF-8 bytes)
    weights (float64, row-major):
        emb:           V × D
        per layer L:
            qkv:       3D × D
            out:       D × D
            fi:        2F × D
            fo:        D × F
        head:          V × D
    optional erase lists (int32 has_erase, then per-layer attn/ffn lists)
    optional tie-break overrides (int32 has_tb, then L × H × int32)

A weight is "zero" iff its float64 bit pattern is exactly +0.0 or -0.0.
The analytically-constructed projections drop entries to literal zero, so
that's what the C++ ``SparseMatrix::build()`` test compares against (see
``hull_cache.py`` / the SparseMatrix definition shared with transformer.cpp).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


def _read_header(f) -> tuple[int, int, int, int, int, int]:
    raw = f.read(24)
    if len(raw) != 24:
        raise ValueError("truncated header")
    return struct.unpack("<6i", raw)


def _skip_vocab(f, V: int) -> None:
    for _ in range(V):
        (n,) = struct.unpack("<I", f.read(4))
        f.read(n)


def _read_matrix(f, rows: int, cols: int) -> np.ndarray:
    n = rows * cols
    buf = f.read(n * 8)
    if len(buf) != n * 8:
        raise ValueError(f"truncated weights: wanted {n*8} bytes, got {len(buf)}")
    return np.frombuffer(buf, dtype=np.float64).reshape(rows, cols)


def _nnz_stats(name: str, w: np.ndarray) -> dict:
    nnz = int(np.count_nonzero(w))
    total = int(w.size)
    return {
        "name": name,
        "shape": list(w.shape),
        "total": total,
        "nnz": nnz,
        "sparsity": 1.0 - nnz / total if total else 0.0,
    }


def measure(path: Path) -> dict:
    """Return a structured report of sparsity per projection per layer."""
    with path.open("rb") as f:
        V, D, L, H, F, stop = _read_header(f)
        _skip_vocab(f, V)

        emb = _read_matrix(f, V, D)
        layers = []
        for li in range(L):
            qkv = _read_matrix(f, 3 * D, D)
            out = _read_matrix(f, D, D)
            fi = _read_matrix(f, 2 * F, D)
            fo = _read_matrix(f, D, F)
            layers.append(
                {
                    "layer": li,
                    "projections": {
                        "qkv": _nnz_stats("qkv", qkv),
                        "out": _nnz_stats("out", out),
                        "fi": _nnz_stats("fi", fi),
                        "fo": _nnz_stats("fo", fo),
                    },
                }
            )
        head = _read_matrix(f, V, D)

    # Aggregate per-layer projection sparsity (the number that determines
    # which row of the issue's table applies).
    proj_total = 0
    proj_nnz = 0
    for L_ in layers:
        for p in L_["projections"].values():
            proj_total += p["total"]
            proj_nnz += p["nnz"]

    return {
        "header": {"V": V, "D": D, "L": L, "H": H, "F": F, "stop": stop},
        "embedding": _nnz_stats("emb", emb),
        "head": _nnz_stats("head", head),
        "layers": layers,
        "projections_total": {
            "total": proj_total,
            "nnz": proj_nnz,
            "sparsity": 1.0 - proj_nnz / proj_total if proj_total else 0.0,
        },
    }


def _fmt_pct(x: float) -> str:
    return f"{x*100:5.1f}%"


def print_report(report: dict, model_path: Path) -> None:
    h = report["header"]
    print(f"\nmodel: {model_path}")
    print(
        f"header: V={h['V']} D={h['D']} L={h['L']} H={h['H']} F={h['F']} "
        f"stop={h['stop']}"
    )

    e = report["embedding"]
    hd = report["head"]
    print(
        f"\n  embedding   shape={tuple(e['shape'])!s:<14}  "
        f"nnz={e['nnz']:>10,} / {e['total']:>10,}   sparsity={_fmt_pct(e['sparsity'])}"
    )
    print(
        f"  head        shape={tuple(hd['shape'])!s:<14}  "
        f"nnz={hd['nnz']:>10,} / {hd['total']:>10,}   sparsity={_fmt_pct(hd['sparsity'])}"
    )

    print(
        f"\n  {'layer':>5}  {'proj':<5}  {'shape':<14}  "
        f"{'nnz':>10}  {'total':>10}  sparsity"
    )
    for L_ in report["layers"]:
        for proj in ("qkv", "out", "fi", "fo"):
            p = L_["projections"][proj]
            print(
                f"  {L_['layer']:>5}  {proj:<5}  {str(tuple(p['shape'])):<14}  "
                f"{p['nnz']:>10,}  {p['total']:>10,}  {_fmt_pct(p['sparsity'])}"
            )

    pt = report["projections_total"]
    print(
        f"\n  per-layer projections combined: "
        f"nnz={pt['nnz']:,} / {pt['total']:,}   sparsity={_fmt_pct(pt['sparsity'])}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure per-projection sparsity of a model.bin"
    )
    ap.add_argument("model", type=Path, help="Path to model.bin")
    ap.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of a table"
    )
    ap.add_argument(
        "--tsv",
        type=Path,
        default=None,
        help="Optional path: write a per-layer-per-projection TSV row dump",
    )
    args = ap.parse_args()

    report = measure(args.model)

    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print_report(report, args.model)

    if args.tsv is not None:
        args.tsv.parent.mkdir(parents=True, exist_ok=True)
        with args.tsv.open("w") as f:
            f.write("layer\tprojection\trows\tcols\tnnz\ttotal\tsparsity\n")
            for L_ in report["layers"]:
                for proj in ("qkv", "out", "fi", "fo"):
                    p = L_["projections"][proj]
                    rows, cols = p["shape"]
                    f.write(
                        f"{L_['layer']}\t{proj}\t{rows}\t{cols}\t"
                        f"{p['nnz']}\t{p['total']}\t{p['sparsity']:.6f}\n"
                    )
        print(f"\n  → per-projection TSV written to {args.tsv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
