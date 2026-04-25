#!/usr/bin/env python3
"""summarize_results.py — Combine per-config bench_real TSVs into one
real_model_summary.tsv with sparse-vs-naive and sparse-vs-blas ratios.

Reads results/real_model_<config>_<prog>.tsv files and emits
results/real_model_summary.tsv as a single overview.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

CONFIGS = ["default", "no_reuse", "max_ffn_32"]
PROGS = ["collatz", "mcm", "sudoku"]


def parse_tsv(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#") or line.startswith("binary"):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        binary, n_tok, n_ops, engine_s, tok_s = parts[:5]
        out[binary] = float(tok_s)
    return out


def main() -> int:
    rows: list[dict[str, str | float]] = []
    for cfg in CONFIGS:
        for prog in PROGS:
            path = RESULTS / f"real_model_{cfg}_{prog}.tsv"
            if not path.exists():
                continue
            d = parse_tsv(path)
            naive = d.get("transformer_naive", 0.0)
            blas  = d.get("transformer_blas",  0.0)
            sparse = d.get("transformer_sparse", 0.0)
            rows.append({
                "config": cfg,
                "prog": prog,
                "naive_tok_s":  naive,
                "blas_tok_s":   blas,
                "sparse_tok_s": sparse,
                "sp_vs_naive":  sparse / naive if naive else 0.0,
                "sp_vs_blas":   sparse / blas if blas else 0.0,
            })

    out = RESULTS / "real_model_summary.tsv"
    cols = ["config", "prog", "naive_tok_s", "blas_tok_s", "sparse_tok_s",
            "sp_vs_naive", "sp_vs_blas"]
    with out.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(
                f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                for c in cols
            ) + "\n")
    print(f"Wrote {out}")
    for r in rows:
        print(f"  {r['config']:<11} {r['prog']:<20} "
              f"naive={r['naive_tok_s']:>8,.0f}  blas={r['blas_tok_s']:>8,.0f}  "
              f"sparse={r['sparse_tok_s']:>8,.0f}  "
              f"sp/naive={r['sp_vs_naive']:>5.2f}×  "
              f"sp/blas={r['sp_vs_blas']:>5.2f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
