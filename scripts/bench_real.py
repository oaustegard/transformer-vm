#!/usr/bin/env python3
"""bench_real.py — Run all three engines on a real model.bin + program file.

The synthetic sweep harness (bench.py) generates random weights at fixed
sparsity levels. This one takes a pre-built model.bin (e.g. from
``transformer_vm.build --save-weights=...``) plus a token-prefix program
(produced by ``wasm-compile``) and runs naive / blas / sparse against it,
verifying byte-identical predicted-token streams and reporting tok/s.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from model_flops import load_stats, per_token_flops, per_token_theoretical

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "build"
RESULTS = REPO / "results"

BINARY_TO_ENGINE = {
    "transformer_naive":  "naive",
    "transformer_blas":   "blas",
    "transformer_sparse": "sparse",
}


def run_engine(binary: str, model: Path, prog: Path, max_gen: int) -> dict:
    cmd = [str(BUILD / binary), str(model), "--regen", "--max-gen", str(max_gen), str(prog)]
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
    n_tok = int(m.group(1)) if m else 0
    n_ops = int(m.group(2)) if m else 0
    engine_time = float(m.group(3)) if m else 0.0
    tok_s = float(m.group(4)) if m else 0.0

    return {
        "binary": binary,
        "wall_s": wall,
        "engine_s": engine_time,
        "n_tok": n_tok,
        "n_ops": n_ops,
        "tok_s": tok_s,
        "stdout": out,
        "stderr": proc.stderr,
        "rc": proc.returncode,
    }


def predicted_tokens(prog: Path) -> list[str]:
    ref = prog.with_name(prog.stem + "_ref.txt")
    return ref.read_text().split()


def fmt(d: dict) -> str:
    extra = ""
    if "gflops_effective" in d:
        # GF/s eff = effective work per token × tok/s. The substrate-fair
        # axis: how productively the engine converts compute time into
        # useful FLOPs (issue #9).
        extra = (
            f"  {d['gflops_effective']:>8.6f} GF/tok  "
            f"{d['effective_gflops_per_s']:>6.2f} GF/s"
        )
    return (
        f"  {d['binary']:<22}  "
        f"{d['n_tok']:>6} tok  "
        f"{d['engine_s']:>7.3f}s  "
        f"{d['tok_s']:>10,.0f} tok/s"
        + extra
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--prog", required=True, type=Path)
    ap.add_argument("--max-gen", type=int, default=20000,
                    help="Cap on generated tokens; engine stops at the program's own halt token if reached first")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--label", default="real",
                    help="Label written into the TSV filename")
    args = ap.parse_args()

    if not args.model.exists():
        sys.exit(f"missing model: {args.model}")
    if not args.prog.exists():
        sys.exit(f"missing program: {args.prog}")

    binaries = ["transformer_naive", "transformer_blas", "transformer_sparse"]
    for b in binaries:
        if not (BUILD / b).exists():
            sys.exit(f"missing binary: {BUILD / b}. Build first.")

    # Introspect the model once for FLOP estimates (#9). prompt_len defaults
    # to 0 here — bench_real reads token streams from disk and we don't know
    # the prefix length without parsing the .txt; the head amortization error
    # from this is small for long runs.
    stats = load_stats(args.model)
    prog_text = args.prog.read_text().split()
    prompt_len = len(prog_text)

    RESULTS.mkdir(exist_ok=True)
    print(f"\nmodel:   {args.model}  (V={stats.shape.V} D={stats.shape.D} "
          f"L={stats.shape.L} H={stats.shape.H} F={stats.shape.F})")
    print(f"  proj nnz {stats.total_proj_nnz:,}/{stats.total_proj_dense:,} "
          f"({100*(1 - stats.total_proj_nnz/stats.total_proj_dense):.1f}% sparse), "
          f"head nnz {stats.head_nnz:,}/{stats.shape.V*stats.shape.D:,} "
          f"({100*(1 - stats.head_nnz/(stats.shape.V*stats.shape.D)):.1f}% sparse)")
    print(f"program: {args.prog} ({prompt_len} prompt tokens)")
    print(f"max-gen: {args.max_gen}    repeats: {args.repeats}\n")

    best_per_bin: dict[str, dict] = {}
    ref_tokens_per_bin: dict[str, list[str]] = {}

    for b in binaries:
        best = None
        # Run --regen `repeats` times on a binary; capture its predicted ref
        # immediately after the *best* run so the saved ref is from the
        # binary we're keeping numbers for.
        best_ref: list[str] = []
        for i in range(args.repeats):
            r = run_engine(b, args.model, args.prog, args.max_gen)
            if r["rc"] != 0:
                print(f"  ✗ {b} returned non-zero rc={r['rc']}")
                print(r["stderr"][-500:])
                return 1
            if best is None or r["engine_s"] < best["engine_s"]:
                best = r
                best_ref = predicted_tokens(args.prog)
        # Annotate with FLOPs columns (issue #9).
        engine = BINARY_TO_ENGINE[b]
        eff = per_token_flops(stats, engine, n_tok=best["n_tok"],
                               prompt_len=prompt_len)
        theo = per_token_theoretical(stats, n_tok=best["n_tok"],
                                      prompt_len=prompt_len)
        best["gflops_effective"]      = eff["total"] / 1e9
        best["gflops_theoretical"]    = theo["total"] / 1e9
        best["effective_gflops_per_s"] = best["gflops_effective"] * best["tok_s"]
        best["tokens_per_gflop"]      = (
            1.0 / best["gflops_effective"] if best["gflops_effective"] > 0 else 0.0
        )
        best_per_bin[b] = best
        ref_tokens_per_bin[b] = best_ref
        # Stash ref under a per-binary name so subsequent --regen overwrites
        # don't clobber what we just captured.
        backup = args.prog.with_name(f"{args.prog.stem}_ref_{b}.txt")
        shutil.copy(args.prog.with_name(args.prog.stem + "_ref.txt"), backup)
        print(fmt(best))

    ref_naive = ref_tokens_per_bin["transformer_naive"]
    print()
    for b in binaries[1:]:
        a = ref_tokens_per_bin[b]
        if a == ref_naive:
            print(f"  ✓ {b} matches naive output exactly ({len(a)} tokens)")
        else:
            mm = next(
                (i for i in range(min(len(a), len(ref_naive))) if a[i] != ref_naive[i]),
                None,
            )
            if mm is None:
                print(f"  ⚠ {b} differs in length: {len(a)} vs {len(ref_naive)}")
            else:
                print(
                    f"  ⚠ {b} first divergence at tok {mm}: "
                    f"{a[mm]!r} vs {ref_naive[mm]!r} (len {len(a)} vs {len(ref_naive)})"
                )

    naive_r = best_per_bin["transformer_naive"]
    blas_r  = best_per_bin["transformer_blas"]
    sparse_r = best_per_bin["transformer_sparse"]
    print(f"\n  → sparse vs naive (tok/s): {sparse_r['tok_s']/naive_r['tok_s']:.2f}×")
    print(f"  → sparse vs BLAS  (tok/s): {sparse_r['tok_s']/blas_r['tok_s']:.2f}×")
    print(f"  → sparse hardware utilization vs BLAS: "
          f"{sparse_r['effective_gflops_per_s']/blas_r['effective_gflops_per_s']:.2f}×  "
          f"(sparse {sparse_r['effective_gflops_per_s']:.2f} GF/s vs "
          f"BLAS {blas_r['effective_gflops_per_s']:.2f} GF/s)")

    out_tsv = RESULTS / f"real_model_{args.label}.tsv"
    cols = ["binary", "n_tok", "n_ops", "engine_s", "tok_s",
            "gflops_theoretical", "gflops_effective",
            "effective_gflops_per_s", "tokens_per_gflop"]
    with out_tsv.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for b in binaries:
            r = best_per_bin[b]
            f.write("\t".join(
                f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c])
                for c in cols
            ) + "\n")
        f.write(
            f"# sparse_vs_naive\t{sparse_r['tok_s']/naive_r['tok_s']:.4f}\t"
            f"sparse_vs_blas\t{sparse_r['tok_s']/blas_r['tok_s']:.4f}\t"
            f"identical_streams\t{ref_tokens_per_bin['transformer_blas'] == ref_naive and ref_tokens_per_bin['transformer_sparse'] == ref_naive}\n"
        )
    print(f"\n  → results written to {out_tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
