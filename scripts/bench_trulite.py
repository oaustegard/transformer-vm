#!/usr/bin/env python3
"""bench_trulite.py — Engine-matrix sweep for issue #18.

For each (binary, program), records:
  - sequential generation tok/s (the per-token forward pass)
  - batched verification tok/s (dgemm + parallel hull, optional head bypass)
  - speedup ratio
  - correctness vs reference

Usage: python3 scripts/bench_trulite.py --model model.bin --out results/trulite_sweep.tsv
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "build"
DATA = REPO / "transformer_vm" / "data"

# 4 binary variants
#  naive    — pre-#5 baseline (scalar matvec, sequential hull, no batched verify path active)
#  sparse   — CSR matvec on sparse projections
#  blas     — full trulite stack: dgemv per token + dgemm batched + parallel hull + head bypass
#  blas_nobypass — ablation: same as blas but head_type ignored (every head goes through the hull)
BINARIES = ["transformer_naive", "transformer_sparse", "transformer_blas", "transformer_blas_nobypass"]

# Token-level programs in increasing length. Sudoku (~1M) is excluded by default
# to keep the matrix tractable; pass --include-sudoku to add it.
DEFAULT_PROGS = ["hello", "addition", "fibonacci", "collatz", "min_cost_matching"]

REGEN_RE = re.compile(
    r"(?:PASS|RAN|REGEN)\s+(\d+)\s+tok(?:/\d+)?(?:\s+\(ref \d+\))?,\s+(\d+)\s+ops in\s+([\d.]+)s\s+\(([\d.]+)\s+tok/s\)"
)
BATCHED_TOTAL_RE = re.compile(r"Batched total:\s+([\d.]+)s\s+\(\s*([\d.]+)\s+tok/s\)")
BATCHED_SPOT_RE = re.compile(r"Spot check:\s+(\S+)")
BATCHED_SPEEDUP_RE = re.compile(r"Speedup vs sequential:\s+([\d.]+)x")


def run_one(binary: str, model: Path, prog: Path, omp_threads: int = 4, blas_threads: int = 1) -> dict:
    cmd = [str(BUILD / binary), str(model), str(prog)]
    env = {
        "OPENBLAS_NUM_THREADS": str(blas_threads),
        "OMP_NUM_THREADS": str(omp_threads),
        "MKL_NUM_THREADS": "1",
        "PATH": "/usr/bin:/bin",
    }
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    wall = time.time() - t0
    out = proc.stdout

    seq_match = REGEN_RE.search(out)
    bt_match = BATCHED_TOTAL_RE.search(out)
    spot_match = BATCHED_SPOT_RE.search(out)
    speedup_match = BATCHED_SPEEDUP_RE.search(out)
    pass_match = "PASS" in out and "FAIL" not in out

    return {
        "binary": binary,
        "prog": prog.stem,
        "wall_s": wall,
        "passed": pass_match and proc.returncode == 0,
        "seq_tok": int(seq_match.group(1)) if seq_match else 0,
        "seq_ops": int(seq_match.group(2)) if seq_match else 0,
        "seq_s": float(seq_match.group(3)) if seq_match else 0.0,
        "seq_tok_s": float(seq_match.group(4)) if seq_match else 0.0,
        "bt_s": float(bt_match.group(1)) if bt_match else 0.0,
        "bt_tok_s": float(bt_match.group(2)) if bt_match else 0.0,
        "bt_speedup": float(speedup_match.group(1)) if speedup_match else 0.0,
        "bt_spot_ok": (spot_match.group(1) == "OK") if spot_match else None,
        "rc": proc.returncode,
        "stderr_tail": proc.stderr[-300:] if proc.stderr else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.bin")
    ap.add_argument("--out", default="results/trulite_sweep.tsv")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--include-sudoku", action="store_true")
    ap.add_argument("--omp-threads", type=int, default=4)
    args = ap.parse_args()

    model = Path(args.model).resolve()
    progs = list(DEFAULT_PROGS)
    if args.include_sudoku:
        progs.append("sudoku")

    rows = []
    print(f"{'binary':<28} {'prog':<20} {'seq tok':>8} {'seq tok/s':>11} {'bt tok/s':>11} {'speedup':>8} {'spot':>5} {'pass':>5}")
    print("-" * 100)
    for prog_name in progs:
        prog = DATA / f"{prog_name}.txt"
        if not prog.exists():
            print(f"  SKIP: {prog} not found")
            continue
        for binary in BINARIES:
            best = None
            for _ in range(args.repeats):
                r = run_one(binary, model, prog, omp_threads=args.omp_threads)
                if not r["passed"] and binary == "transformer_naive":
                    # naive is the reference — if it fails, surface it
                    print(f"  WARN: naive failed on {prog_name}: rc={r['rc']}")
                    print(f"    stderr: {r['stderr_tail']}")
                if best is None or (r["seq_s"] > 0 and r["seq_s"] < best["seq_s"]):
                    best = r
            rows.append(best)
            spot = "OK" if best["bt_spot_ok"] else ("BAD" if best["bt_spot_ok"] is False else "—")
            passed = "OK" if best["passed"] else "FAIL"
            print(f"{binary:<28} {prog_name:<20} {best['seq_tok']:>8} {best['seq_tok_s']:>11,.0f} "
                  f"{best['bt_tok_s']:>11,.0f} {best['bt_speedup']:>7.2f}x {spot:>5} {passed:>5}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["binary", "prog", "seq_tok", "seq_ops", "seq_s", "seq_tok_s",
            "bt_s", "bt_tok_s", "bt_speedup", "bt_spot_ok", "passed", "rc"]
    with out.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    sys.exit(main())
