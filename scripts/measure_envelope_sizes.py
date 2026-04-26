#!/usr/bin/env python3
"""measure_envelope_sizes.py — Issue #7 step 1.

Runs the PROFILE_PHASES sparse engine on real model.bin files across configs
and programs, captures per-(layer, head) final envelope sizes from stderr,
and writes one TSV per (config, prog) plus a combined histogram summary.

Each ENVELOPE line emitted by the engine has the form:
    ENVELOPE\\t<test>\\t<layer>\\t<head>\\t<upper_size>\\t<lower_size>

The decision criterion is in the issue text:
  - 90 %+ of layer-heads ending with envelope < ~50 lines  -> swap is worth it.
  - Most ending > 200                                       -> skip.

Usage:
    uv run python scripts/measure_envelope_sizes.py \\
        --models models --progs data --max-gen 20000
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "build"
RESULTS = REPO / "results"


def parse_envelope_lines(stderr: str) -> list[tuple[str, int, int, int, int]]:
    rows = []
    for ln in stderr.splitlines():
        if not ln.startswith("ENVELOPE\t"):
            continue
        parts = ln.split("\t")
        # ENVELOPE, test, layer, head, upper, lower
        if len(parts) != 6:
            continue
        rows.append((parts[1], int(parts[2]), int(parts[3]),
                     int(parts[4]), int(parts[5])))
    return rows


def run_one(binary: Path, model: Path, prog: Path, max_gen: int, timeout: int) -> dict:
    cmd = [str(binary), str(model), "--regen",
           f"--max-gen={max_gen}", str(prog)]
    env = {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PATH": "/usr/bin:/bin",
    }
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, env=env)
    wall = time.time() - t0
    return {
        "rc": proc.returncode,
        "wall_s": wall,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "rows": parse_envelope_lines(proc.stderr),
    }


def histogram(values: list[int], buckets: tuple[int, ...]) -> list[tuple[str, int, float]]:
    out = []
    n = len(values)
    if n == 0:
        return out
    prev = 0
    for b in buckets:
        c = sum(1 for v in values if prev < v <= b)
        out.append((f"{prev+1}..{b}", c, c / n))
        prev = b
    c = sum(1 for v in values if v > prev)
    out.append((f">{prev}", c, c / n))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=REPO / "models")
    ap.add_argument("--progs",  type=Path, default=REPO / "data")
    ap.add_argument("--binary", default="transformer_sparse_prof")
    ap.add_argument("--max-gen", type=int, default=20000)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--configs", nargs="*",
                    default=["default", "no_reuse", "max_ffn_32"])
    ap.add_argument("--progs-list", nargs="*",
                    default=["collatz", "min_cost_matching", "sudoku"])
    args = ap.parse_args()

    binary = BUILD / args.binary
    if not binary.exists():
        print(f"missing binary: {binary}")
        print("hint: run `make profile` to build the *_prof binaries")
        return 1

    RESULTS.mkdir(exist_ok=True)
    summary_rows = []
    BUCKETS = (10, 25, 50, 100, 200, 500, 1000, 5000)

    for cfg in args.configs:
        model = args.models / f"{cfg}.bin"
        if not model.exists():
            print(f"  ✗ missing model {model} — skip"); continue
        for prog_name in args.progs_list:
            prog = args.progs / f"{prog_name}.txt"
            if not prog.exists():
                print(f"  ✗ missing prog {prog} — skip"); continue

            print(f"  ▶ {cfg:<11} {prog_name:<20} max_gen={args.max_gen}")
            r = run_one(binary, model, prog, args.max_gen, args.timeout)
            if r["rc"] != 0:
                print(f"    rc={r['rc']}  stderr tail:\n{r['stderr'][-400:]}")
                continue

            rows = r["rows"]
            if not rows:
                print(f"    no ENVELOPE lines emitted; skip")
                continue

            out_tsv = RESULTS / f"envelope_sizes_{cfg}_{prog_name}.tsv"
            with out_tsv.open("w") as f:
                f.write("test\tlayer\thead\tupper\tlower\n")
                for (t, l, h, u, lo) in rows:
                    f.write(f"{t}\t{l}\t{h}\t{u}\t{lo}\n")
            print(f"    → {out_tsv} ({len(rows)} rows, wall={r['wall_s']:.1f}s)")

            uppers = [u for (_, _, _, u, _)  in rows]
            lowers = [lo for (_, _, _, _, lo) in rows]
            both   = uppers + lowers
            for label, vals in [("upper", uppers), ("lower", lowers), ("both", both)]:
                h = histogram(vals, BUCKETS)
                line = f"      {label:<5} n={len(vals):>4}  " + \
                       "  ".join(f"{b}:{c}({p*100:.0f}%)" for (b, c, p) in h)
                print(line)
                # Aggregate into summary
                for (bucket, count, frac) in h:
                    summary_rows.append({
                        "config": cfg, "prog": prog_name, "side": label,
                        "bucket": bucket, "count": count, "frac": frac,
                        "n": len(vals),
                    })

    summary_tsv = RESULTS / "envelope_sizes_summary.tsv"
    with summary_tsv.open("w") as f:
        f.write("config\tprog\tside\tbucket\tcount\tfrac\tn\n")
        for r in summary_rows:
            f.write(f"{r['config']}\t{r['prog']}\t{r['side']}\t{r['bucket']}\t{r['count']}\t{r['frac']:.4f}\t{r['n']}\n")
    print(f"\n  → {summary_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
