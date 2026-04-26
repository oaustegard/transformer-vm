#!/usr/bin/env python3
"""Augment results/trulite_sweep.tsv with FLOPs-normalized columns.

For each (binary, prog) row, compute:
  - gflops_per_token (engine-aware: sparse uses nnz, naive/blas use dense)
  - effective_gflops_seq = gflops_per_token * seq_tok_s
  - effective_gflops_bt  = gflops_per_token * bt_tok_s

The dgemm-batched path executes the *same FLOP count* as the per-token
dense path (it's the same matmul, just batched), so blas and blas_nobypass
share the dense FLOP count for both seq and bt rates. Head bypass changes
the attention FLOP count slightly (passthrough/gather are O(1) instead of
O(log n)) but the proj cost dominates, so it's a wash in this metric.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from model_flops import load_stats, per_token_flops  # noqa: E402

BINARY_TO_ENGINE = {
    "transformer_naive":          "naive",
    "transformer_sparse":         "sparse",
    "transformer_blas":           "blas",
    "transformer_blas_nobypass":  "blas",
}


def main():
    src = REPO / "results" / "trulite_sweep.tsv"
    dst = REPO / "results" / "trulite_sweep_flops.tsv"
    model = REPO / "model.bin"
    stats = load_stats(model)

    with src.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    out_cols = list(reader.fieldnames) + [
        "gf_per_tok", "eff_gflops_seq", "eff_gflops_bt"
    ]
    with dst.open("w") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            engine = BINARY_TO_ENGINE[r["binary"]]
            n_tok = int(r["seq_tok"])
            ft = per_token_flops(stats, engine, n_tok=n_tok, prompt_len=0)
            gf = ft["total"] / 1e9
            r["gf_per_tok"] = f"{gf:.6f}"
            seq_ts = float(r["seq_tok_s"])
            bt_ts = float(r["bt_tok_s"])
            r["eff_gflops_seq"] = f"{gf * seq_ts:.4f}"
            r["eff_gflops_bt"] = f"{gf * bt_ts:.4f}"
            w.writerow(r)
    print(f"Wrote {dst}")

    # Pretty-print
    print(f"\n{'binary':<28} {'prog':<20} {'GF/tok':>10} {'GF/s seq':>10} {'GF/s bt':>10}")
    print("-" * 84)
    for r in rows:
        print(f"{r['binary']:<28} {r['prog']:<20} {r['gf_per_tok']:>10} "
              f"{r['eff_gflops_seq']:>10} {r['eff_gflops_bt']:>10}")


if __name__ == "__main__":
    main()
