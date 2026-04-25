#!/usr/bin/env python3
"""analyze_erase_propagation.py — Quantify the CSR-pruning headroom from
propagating the model's erase lists through the projection pipeline.

The Python builder (``transformer_vm/model/weights.py``) emits two
per-layer lists:

    attn_erase[L] — residual-slot indices zeroed at end of layer L's
                    attention block.
    ffn_erase[L]  — residual-slot indices zeroed at end of layer L's FFN.

These slots are written into ``model.bin`` and read into ``Model`` by
``transformer.cpp::load()``, but the inference loop never consumes them.
A slot known to hold zero at the input of a projection means that
projection's matvec multiplies a zero into every CSR entry whose column
matches that slot — work that can be skipped if the projection's CSR
representation is pruned at load time.

This script simulates the propagation across half-layers and reports,
per projection, how many CSR entries point at always-zero input
columns (i.e. would be droppable). It does NOT modify the engine — it
only quantifies the upside before that work is done.

Propagation rule (one full layer):

    prune qkv[L] columns in zero_set
    zero_set ← (zero_set − nonzero_rows(out[L])) ∪ attn_erase[L]
    prune fi[L]  columns in zero_set
    zero_set ← (zero_set − nonzero_rows(fo[L]))  ∪ ffn_erase[L]
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def _read_header(f) -> tuple[int, int, int, int, int, int]:
    return struct.unpack("<6i", f.read(24))


def _skip_vocab(f, V: int) -> None:
    for _ in range(V):
        (n,) = struct.unpack("<I", f.read(4))
        f.read(n)


def _read_matrix(f, rows: int, cols: int) -> np.ndarray:
    n = rows * cols
    return np.frombuffer(f.read(n * 8), dtype=np.float64).reshape(rows, cols)


def load_model(path: Path) -> dict:
    """Parse model.bin into a dict the propagation walk can consume."""
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
            layers.append({"qkv": qkv, "out": out, "fi": fi, "fo": fo})
        head = _read_matrix(f, V, D)

        attn_erase: list[list[int]] = [[] for _ in range(L)]
        ffn_erase: list[list[int]] = [[] for _ in range(L)]
        raw = f.read(4)
        if len(raw) == 4:
            (has_erase,) = struct.unpack("<i", raw)
            if has_erase:
                for li in range(L):
                    (n,) = struct.unpack("<i", f.read(4))
                    attn_erase[li] = list(struct.unpack(f"<{n}i", f.read(4 * n)))
                    (n,) = struct.unpack("<i", f.read(4))
                    ffn_erase[li] = list(struct.unpack(f"<{n}i", f.read(4 * n)))

    return {
        "header": {"V": V, "D": D, "L": L, "H": H, "F": F, "stop": stop},
        "embedding": emb,
        "head": head,
        "layers": layers,
        "attn_erase": attn_erase,
        "ffn_erase": ffn_erase,
    }


def nonzero_rows(W: np.ndarray) -> set[int]:
    """Rows of W with at least one nonzero entry — the residual dims this
    projection actually writes to."""
    return set(int(i) for i in np.where((W != 0).any(axis=1))[0])


def csr_columns(W: np.ndarray) -> np.ndarray:
    """Column index of every nonzero entry in W, flat."""
    return np.where(W != 0)[1]


def prune_count(W: np.ndarray, zero_cols: set[int]) -> tuple[int, int]:
    """Return (droppable_nnz, total_nnz) for projection W given that
    columns in ``zero_cols`` are guaranteed zero in the input vector."""
    cols = csr_columns(W)
    total = int(cols.size)
    if not zero_cols or total == 0:
        return 0, total
    mask = np.isin(cols, list(zero_cols))
    return int(mask.sum()), total


def analyze(model: dict) -> dict:
    """Walk the half-layer pipeline; report per-projection prunability."""
    L = model["header"]["L"]
    D = model["header"]["D"]

    # Initial zero_set: nothing assumed about input residual on layer 0.
    # (Embedding writes to all D dims unless its output is sparse, but
    # the input projection of layer 0 runs against the token embedding +
    # position encoding, so we conservatively start with zero_set = {}.)
    zero_set: set[int] = set()

    # For comparison: also track the trivially-true zero set (slots no
    # projection ever writes nonzero — but those won't show up because
    # something always writes them, otherwise the slot wouldn't exist).

    rows = []
    grand_droppable = 0
    grand_total = 0

    for li in range(L):
        ly = model["layers"][li]

        # qkv[li] reads residual at input of layer li
        d_qkv, t_qkv = prune_count(ly["qkv"], zero_set)
        rows.append({"layer": li, "proj": "qkv", "in_zero_cols": len(zero_set),
                     "droppable_nnz": d_qkv, "total_nnz": t_qkv,
                     "shape": ly["qkv"].shape})
        grand_droppable += d_qkv
        grand_total += t_qkv

        # End of attention: out writes nonzero rows; attn_erase zeroes its slots.
        out_writes = nonzero_rows(ly["out"])
        zero_set = (zero_set - out_writes) | set(model["attn_erase"][li])

        # fi[li] reads residual at this point
        d_fi, t_fi = prune_count(ly["fi"], zero_set)
        rows.append({"layer": li, "proj": "fi", "in_zero_cols": len(zero_set),
                     "droppable_nnz": d_fi, "total_nnz": t_fi,
                     "shape": ly["fi"].shape})
        grand_droppable += d_fi
        grand_total += t_fi

        # End of FFN: fo writes nonzero rows; ffn_erase zeroes its slots.
        fo_writes = nonzero_rows(ly["fo"])
        zero_set = (zero_set - fo_writes) | set(model["ffn_erase"][li])

        # `out` and `fo` projections also read but don't take residual as
        # input — qkv produces an attention-attended vector that feeds out;
        # fi produces the FFN hidden that feeds fo. Their input space is
        # intra-block, not the residual stream, so erase-set propagation
        # doesn't directly apply. They benefit only from their own row
        # sparsity, which CSR already exploits.

    return {
        "header": model["header"],
        "rows": rows,
        "grand_droppable": grand_droppable,
        "grand_total": grand_total,
        "final_zero_set_size": len(zero_set),
        "D": D,
    }


def print_report(rep: dict, model_path: Path) -> None:
    h = rep["header"]
    print(f"\nmodel: {model_path}")
    print(f"header: V={h['V']} D={h['D']} L={h['L']} H={h['H']} F={h['F']}\n")

    print(
        f"  {'layer':>5}  {'proj':<4}  {'shape':<14}  "
        f"{'in_zero':>7}  {'droppable':>9}  {'total':>7}  reduction"
    )
    for r in rep["rows"]:
        red = (r["droppable_nnz"] / r["total_nnz"] * 100) if r["total_nnz"] else 0.0
        print(
            f"  {r['layer']:>5}  {r['proj']:<4}  {str(tuple(r['shape'])):<14}  "
            f"{r['in_zero_cols']:>7}  {r['droppable_nnz']:>9}  "
            f"{r['total_nnz']:>7}  {red:>6.1f}%"
        )

    g = (
        rep["grand_droppable"] / rep["grand_total"] * 100
        if rep["grand_total"]
        else 0.0
    )
    print(
        f"\n  total: {rep['grand_droppable']:,} droppable / "
        f"{rep['grand_total']:,} CSR nnz across qkv+fi  →  {g:.2f}% reduction"
    )
    print(
        f"  final zero_set size: {rep['final_zero_set_size']} / D={rep['D']} "
        f"({rep['final_zero_set_size']/rep['D']*100:.1f}% of residual stream)"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument(
        "--tsv",
        type=Path,
        default=None,
        help="Optional: write per-projection rows as TSV",
    )
    args = ap.parse_args()

    model = load_model(args.model)
    rep = analyze(model)
    print_report(rep, args.model)

    if args.tsv is not None:
        args.tsv.parent.mkdir(parents=True, exist_ok=True)
        with args.tsv.open("w") as f:
            f.write(
                "layer\tproj\trows\tcols\tin_zero_cols\tdroppable_nnz\t"
                "total_nnz\treduction\n"
            )
            for r in rep["rows"]:
                rows_, cols_ = r["shape"]
                red = (
                    r["droppable_nnz"] / r["total_nnz"]
                    if r["total_nnz"]
                    else 0.0
                )
                f.write(
                    f"{r['layer']}\t{r['proj']}\t{rows_}\t{cols_}\t"
                    f"{r['in_zero_cols']}\t{r['droppable_nnz']}\t"
                    f"{r['total_nnz']}\t{red:.6f}\n"
                )
        print(f"\n  → TSV written to {args.tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
