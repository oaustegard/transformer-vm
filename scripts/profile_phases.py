#!/usr/bin/env python3
"""profile_phases.py — Run the -DPROFILE_PHASES sparse engine on real model.bin
files across configs and programs, capture phase breakdowns.

Sister to bench_real.py, but for the profile-build binaries that emit the
finer-grained timing block (embed/qkv/attn/out/ffn/head/misc).

Usage:
    uv run python scripts/profile_phases.py --models models --progs data
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "build"
RESULTS = REPO / "results"

PHASES = ("embed", "qkv", "attn", "out", "ffn", "head", "misc")


def run(binary: str, model: Path, prog: Path, max_gen: int) -> dict:
    cmd = [str(BUILD / binary), str(model), "--regen",
           f"--max-gen={max_gen}", str(prog)]
    env = {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PATH": "/usr/bin:/bin",
    }
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    wall = time.time() - t0
    out = proc.stdout

    m = re.search(
        r"REGEN\s+(\d+)\s+tok,\s+(\d+)\s+ops in\s+([\d.]+)s\s+\(([\d.]+)\s+tok/s\)", out
    )
    if not m:
        return {"binary": binary, "rc": proc.returncode, "wall_s": wall,
                "stderr": proc.stderr[-500:]}

    res = {
        "binary": binary,
        "rc": proc.returncode,
        "wall_s": wall,
        "n_tok": int(m.group(1)),
        "n_ops": int(m.group(2)),
        "engine_s": float(m.group(3)),
        "tok_s": float(m.group(4)),
    }

    for phase in PHASES:
        pm = re.search(rf"{phase}:\s+([\d.]+)s\s+\(\s*([\d.]+)%\)", out)
        res[f"{phase}_s"] = float(pm.group(1)) if pm else 0.0
        res[f"{phase}_pct"] = float(pm.group(2)) if pm else 0.0

    pm = re.search(r"proj:\s+([\d.]+)s\s+\(\s*([\d.]+)%\)", out)
    res["proj_s"] = float(pm.group(1)) if pm else 0.0
    res["proj_pct"] = float(pm.group(2)) if pm else 0.0
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=REPO / "models")
    ap.add_argument("--progs", type=Path, default=REPO / "data")
    ap.add_argument("--max-gen", type=int, default=20000)
    ap.add_argument("--label", default="phases")
    args = ap.parse_args()

    configs = [
        ("default",     args.models / "default.bin"),
        ("no_reuse",    args.models / "no_reuse.bin"),
        ("max_ffn_32",  args.models / "max_ffn_32.bin"),
    ]
    progs = [
        ("collatz",            args.progs / "collatz.txt"),
        ("min_cost_matching",  args.progs / "min_cost_matching.txt"),
    ]
    binaries = ["transformer_naive_prof", "transformer_blas_prof", "transformer_sparse_prof"]

    RESULTS.mkdir(exist_ok=True)
    rows: list[dict] = []

    for cfg_name, model in configs:
        if not model.exists():
            print(f"  ✗ missing model: {model} — skip")
            continue
        for prog_name, prog in progs:
            if not prog.exists():
                print(f"  ✗ missing program: {prog} — skip")
                continue
            for b in binaries:
                r = run(b, model, prog, args.max_gen)
                r.update({"config": cfg_name, "prog": prog_name})
                if r.get("rc", 1) != 0 or "n_tok" not in r:
                    print(f"  ✗ {cfg_name}/{prog_name}/{b} failed")
                    print(r.get("stderr", "")[-200:])
                    continue
                rows.append(r)
                print(
                    f"  {cfg_name:<11} {prog_name:<20} {b:<25}"
                    f" {r['tok_s']:>8.0f} tok/s  "
                    + "  ".join(f"{p}={r[f'{p}_pct']:>4.1f}%" for p in PHASES)
                )

    out_tsv = RESULTS / f"phase_breakdown_{args.label}.tsv"
    cols = ["config", "prog", "binary", "n_tok", "engine_s", "tok_s"] + \
           [f"{p}_s" for p in PHASES] + [f"{p}_pct" for p in PHASES] + \
           ["proj_s", "proj_pct"]
    with out_tsv.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(
                f"{r.get(c, ''):.4f}" if isinstance(r.get(c), float) else str(r.get(c, ""))
                for c in cols
            ) + "\n")
    print(f"\n  → results written to {out_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
