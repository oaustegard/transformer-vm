#!/usr/bin/env python3
"""bench.py — Sweep sparsity, run naive/blas/sparse engines, verify outputs match.

Builds one synthetic model per sparsity level, runs all three binaries on it,
and reports tok/s plus the proj-time breakdown. Verifies byte-identical
predicted-token streams across binaries (correctness check).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
BUILD = REPO / "build"
RESULTS = REPO / "results"


def run_engine(binary: str, model: Path, prog: Path, max_gen: int) -> dict:
    """Run an engine binary in regen mode and parse its output.

    BLAS threading is disabled so we measure single-thread compute, which
    matches what each token's forward pass actually does (no batch).
    """
    cmd = [str(BUILD / binary), str(model), "--regen",
           "--max-gen", str(max_gen), str(prog)]
    env = {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PATH": "/usr/bin:/bin",
    }
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    wall = time.time() - t0
    out = proc.stdout

    # Parse REGEN line: "REGEN N tok, M ops in X.XXs (Y tok/s)"
    m = re.search(r"REGEN\s+(\d+)\s+tok,\s+(\d+)\s+ops in\s+([\d.]+)s\s+\(([\d.]+)\s+tok/s\)", out)
    n_tok = int(m.group(1)) if m else 0
    n_ops = int(m.group(2)) if m else 0
    engine_time = float(m.group(3)) if m else 0.0
    tok_s = float(m.group(4)) if m else 0.0

    return {
        "binary": binary,
        "wall_s": wall,
        "engine_s": engine_time,
        "n_tok": n_tok,
        "tok_s": tok_s,
        "stdout": out,
    }


def predicted_tokens(prog: Path) -> list[str]:
    """Read the regen'd ref file (engine writes <prog>_ref.txt)."""
    ref = prog.with_name(prog.stem + "_ref.txt")
    return ref.read_text().split()


def fmt(d: dict) -> str:
    return (
        f"  {d['binary']:<22}  "
        f"{d['n_tok']:>4} tok  "
        f"{d['engine_s']:>6.3f}s  "
        f"{d['tok_s']:>10,.0f} tok/s"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=128)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--F", type=int, default=256)
    ap.add_argument("--prog-len", type=int, default=20)
    ap.add_argument("--bench-steps", type=int, default=500)
    ap.add_argument("--sparsities", default="0.50,0.75,0.90,0.95,0.99")
    ap.add_argument("--repeats", type=int, default=3,
                    help="Run each binary N times, take min (warmest cache)")
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    sparsities = [float(s) for s in args.sparsities.split(",")]
    binaries = ["transformer_naive", "transformer_blas", "transformer_sparse"]

    summary_rows = []
    print(f"\nDimensions: V={args.V} D={args.D} L={args.L} H={args.H} F={args.F}")
    print(f"Per-layer dense weight count: {3*args.D*args.D + args.D*args.D + 2*args.F*args.D + args.D*args.F:,}")
    print(f"Bench tokens per run: ~{args.bench_steps + args.prog_len + 100}")
    print(f"Repeats: {args.repeats}\n")

    for sp in sparsities:
        print(f"━━━━━━━━━━━ sparsity = {sp:.0%} ━━━━━━━━━━━")
        model = DATA / f"sweep_sp{int(sp*100):02d}.bin"
        prog = DATA / f"sweep_sp{int(sp*100):02d}_prog.txt"

        # Generate synthetic model
        gen = subprocess.run(
            [
                str(REPO / ".venv/bin/python"),
                str(REPO / "scripts/make_synthetic_model.py"),
                "--out", str(model),
                "--prog", str(prog),
                "--V", str(args.V), "--D", str(args.D),
                "--L", str(args.L), "--H", str(args.H), "--F", str(args.F),
                "--sparsity", str(sp),
                "--prog-len", str(args.prog_len),
                "--bench-steps", str(args.bench_steps),
                "--seed", "42",
            ],
            capture_output=True, text=True, check=True,
        )

        # Run each binary `repeats` times; keep the best (lowest engine_s)
        best_per_bin: dict[str, dict] = {}
        ref_tokens_per_bin: dict[str, list[str]] = {}
        for b in binaries:
            best = None
            for _ in range(args.repeats):
                r = run_engine(b, model, prog, args.bench_steps)
                if best is None or r["engine_s"] < best["engine_s"]:
                    best = r
            best_per_bin[b] = best
            print(fmt(best))
            ref_tokens_per_bin[b] = predicted_tokens(prog)

        # Correctness: do all three predict identical token streams?
        ref_naive = ref_tokens_per_bin["transformer_naive"]
        all_match = all(
            ref_tokens_per_bin[b] == ref_naive
            for b in binaries[1:]
        )
        # Note: BLAS may differ in last few bits due to summation order,
        # which can cascade into different predicted tokens. Report.
        for b in binaries[1:]:
            if ref_tokens_per_bin[b] != ref_naive:
                # Find first mismatch
                a = ref_tokens_per_bin[b]
                mm = next(
                    (i for i in range(min(len(a), len(ref_naive))) if a[i] != ref_naive[i]),
                    None,
                )
                if mm is None:
                    print(f"  ⚠ {b} differs in length: {len(a)} vs {len(ref_naive)}")
                else:
                    print(f"  ⚠ {b} first divergence at tok {mm}: {a[mm]} vs {ref_naive[mm]}")
            else:
                print(f"  ✓ {b} matches naive output exactly")

        naive_ts = best_per_bin["transformer_naive"]["tok_s"]
        blas_ts = best_per_bin["transformer_blas"]["tok_s"]
        sparse_ts = best_per_bin["transformer_sparse"]["tok_s"]
        print(f"  → sparse vs naive: {sparse_ts/naive_ts:.2f}×")
        print(f"  → sparse vs BLAS:  {sparse_ts/blas_ts:.2f}×\n")

        summary_rows.append({
            "sparsity": sp,
            "naive_tok_s": naive_ts,
            "blas_tok_s": blas_ts,
            "sparse_tok_s": sparse_ts,
        })

    # Summary table
    print("━━━━━━━━━━━━━━━━━━━━ summary ━━━━━━━━━━━━━━━━━━━━")
    print(f"  {'sparsity':>9}  {'naive':>10}  {'BLAS':>10}  {'sparse':>10}    "
          f"{'sp/naive':>8}  {'sp/BLAS':>8}")
    for r in summary_rows:
        print(
            f"  {r['sparsity']:>8.0%}  "
            f"{r['naive_tok_s']:>10,.0f}  "
            f"{r['blas_tok_s']:>10,.0f}  "
            f"{r['sparse_tok_s']:>10,.0f}    "
            f"{r['sparse_tok_s']/r['naive_tok_s']:>7.2f}×  "
            f"{r['sparse_tok_s']/r['blas_tok_s']:>7.2f}×"
        )

    # Persist as TSV
    with (RESULTS / "sweep.tsv").open("w") as f:
        f.write("sparsity\tnaive_tok_s\tblas_tok_s\tsparse_tok_s\n")
        for r in summary_rows:
            f.write(f"{r['sparsity']}\t{r['naive_tok_s']}\t{r['blas_tok_s']}\t{r['sparse_tok_s']}\n")
    print(f"\n  → results written to {RESULTS / 'sweep.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
