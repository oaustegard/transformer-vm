#!/usr/bin/env python3
"""check_determinism.py — Run each engine variant twice on the same model.bin
and program; report whether predicted-token streams are byte-identical
within an engine and whether engines agree with the naive baseline.

Same-engine drift = bug. Cross-engine drift = expected FP-rounding effect
(see issue #10's divergence-horizon characterization).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "build"
DATA = REPO / "transformer_vm" / "data"

BINARIES = ["transformer_naive", "transformer_sparse",
            "transformer_blas", "transformer_blas_nobypass"]


def run_capture_tokens(binary: str, model: Path, prog: Path) -> tuple[bool, list[str], str]:
    """Run with --regen so the engine writes <prog>_ref.txt; read it back."""
    ref_path = prog.with_name(prog.stem + "_ref.txt")
    # Save and restore the ground-truth ref so we don't lose it.
    backup = ref_path.read_text() if ref_path.exists() else None
    try:
        env = {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PATH": "/usr/bin:/bin",
        }
        cmd = [str(BUILD / binary), str(model), "--regen", str(prog)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        if proc.returncode != 0:
            return False, [], proc.stderr[-500:]
        # Engine wrote ref. Read regenerated tokens.
        toks = ref_path.read_text().split()
        return True, toks, ""
    finally:
        if backup is not None:
            ref_path.write_text(backup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.bin")
    ap.add_argument("--prog", default="hello",
                    help="Program name (without .txt)")
    args = ap.parse_args()

    model = Path(args.model).resolve()
    prog = DATA / f"{args.prog}.txt"

    if not prog.exists():
        print(f"FATAL: {prog} not found", file=sys.stderr)
        return 1

    runs: dict[str, list[list[str]]] = {}
    print(f"Determinism check on {prog.name}\n")
    print(f"{'binary':<28}  run1==run2  vs naive")
    print("-" * 64)
    for b in BINARIES:
        runs[b] = []
        for _ in range(2):
            ok, toks, err = run_capture_tokens(b, model, prog)
            if not ok:
                print(f"{b:<28}  FAIL    err: {err[:80]}")
                runs[b].append([])
                continue
            runs[b].append(toks)
        same = (runs[b][0] == runs[b][1] and len(runs[b][0]) > 0)
        runs_str = "OK" if same else "DIFF"
        if "transformer_naive" in runs and runs["transformer_naive"]:
            naive_toks = runs["transformer_naive"][0]
            if not naive_toks or not runs[b][0]:
                vs = "—"
            else:
                # Find first diverging position (or "match" if identical)
                if runs[b][0] == naive_toks:
                    vs = "match"
                else:
                    n = min(len(runs[b][0]), len(naive_toks))
                    div = next((i for i in range(n) if runs[b][0][i] != naive_toks[i]), n)
                    vs = f"div@{div}/{len(naive_toks)}"
        else:
            vs = "—"
        print(f"{b:<28}  {runs_str:<8}  {vs}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
