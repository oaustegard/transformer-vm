#!/usr/bin/env python3
"""divergence_horizon.py — Issue #10 step 1.

Sweeps the FP-rounding divergence horizon for byte-identical token streams
across (program, engine_pair, model_config). The phenomenon: BLAS and naive
engines compute the same dot product up to rounding (vectorized FMA vs
sequential scalar), and over a long enough run that error compounds until
it crosses a decision boundary in head argmax (or attention argmax). It's
deterministic FP, not non-determinism.

For each (program, engine_a, engine_b, model_config):
  1. Run engine_a with --regen, capture its _ref.txt.
  2. Run engine_b same, capture its _ref.txt.
  3. Token-by-token compare to find the first position where the streams
     differ. That position is the divergence horizon.
  4. If divergence found: re-run both engines with --diag-at=<abs_pos> to
     emit per-(layer, head) attn argmax kx-of-winning-key plus head logit
     top-2 at that pos. Compare:
       - any (layer, head) attn winner kx differs -> attn_argmax
       - all attn winners match, head argmax differs -> head_argmax
       - everything matches but streams differ further out -> unknown
         (shouldn't happen with deterministic engines on identical inputs)
  5. If no divergence within --max-gen budget, record first_diverging_token=-1.

Output: results/divergence_horizon.tsv with columns
  program  engine_a  engine_b  model_config
  first_diverging_token  total_tokens_compared  divergence_cause

Note: the `naive` and `sparse` engines share scalar summation order
(sparse iterates only the nonzero columns, in the same order as naive's
dense loop, and `s += 0 * x[j]` is exact), so their streams should be
byte-identical for the full budget. The interesting pair is `blas vs
naive` (and equivalently `blas vs sparse`).

Usage:
    uv run python scripts/divergence_horizon.py \\
        --models models --progs transformer_vm/data --max-gen 10000000
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "build"
RESULTS = REPO / "results"

BINARY_TO_ENGINE = {
    "transformer_naive":  "naive",
    "transformer_blas":   "blas",
    "transformer_sparse": "sparse",
}
ENGINE_TO_BINARY = {v: k for k, v in BINARY_TO_ENGINE.items()}

ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PATH": "/usr/bin:/bin",
}


def run_engine(
    binary_name: str,
    model: Path,
    prog: Path,
    max_gen: int,
    diag_at: int | None = None,
    timeout: int = 1800,
) -> dict:
    """Run an engine binary in --regen mode; optionally request a diag dump."""
    cmd = [str(BUILD / binary_name), str(model), "--regen",
           f"--max-gen={max_gen}", str(prog)]
    if diag_at is not None:
        cmd.append(f"--diag-at={diag_at}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, env=ENV)
    return {
        "rc": proc.returncode,
        "wall_s": time.time() - t0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def read_ref_tokens(prog: Path) -> list[str]:
    ref = prog.with_name(prog.stem + "_ref.txt")
    return ref.read_text().split()


def first_divergence(a: list[str], b: list[str]) -> int:
    """Index of first differing token, or -1 if streams match within min len.

    A length mismatch with otherwise-matching tokens up to min length
    is also a divergence (one engine emitted a stop earlier), and is
    reported at index = min_len.
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return -1


def parse_diag(stderr: str) -> dict | None:
    """Parse the single DIAG line emitted by --diag-at."""
    for ln in stderr.splitlines():
        if not ln.startswith("DIAG\t"):
            continue
        parts = ln.split("\t")
        # DIAG, test, pos, head_best, head_second, head_best_score,
        # head_second_score, then L*H kx values.
        if len(parts) < 7:
            return None
        return {
            "test": parts[1],
            "pos": int(parts[2]),
            "head_best": int(parts[3]),
            "head_second": int(parts[4]),
            "head_best_score": parts[5],   # keep as string for exact compare
            "head_second_score": parts[6],
            "kx": parts[7:],               # strings; bitwise compare
        }
    return None


def attribute_cause(diag_a: dict | None, diag_b: dict | None) -> str:
    """Compare two engines' DIAG lines at the divergence boundary."""
    if diag_a is None or diag_b is None:
        return "unknown"
    if len(diag_a["kx"]) != len(diag_b["kx"]):
        return "unknown"  # different model — shouldn't happen
    # Bitwise compare each (layer, head) winner kx (printed with %.17g
    # so identical doubles print identically).
    for ka, kb in zip(diag_a["kx"], diag_b["kx"]):
        if ka != kb:
            return "attn_argmax"
    if diag_a["head_best"] != diag_b["head_best"]:
        return "head_argmax"
    # All match per-head, head argmax matches — shouldn't happen if the
    # streams actually differ at this pos. Either the divergence index is
    # wrong, or there's an edge case we don't model (e.g., diag fired at
    # the wrong pos because one engine stopped early).
    return "unknown"


def cached_run(
    cache: dict[str, dict],
    engine: str,
    model: Path,
    prog: Path,
    max_gen: int,
    timeout: int,
) -> dict | None:
    """Run an engine once per (engine, model, prog, max_gen) and cache its
    token list. Three-pair sweep on one (model, prog) only does 3 runs,
    not 6. Cache keyed on engine name since model/prog/max-gen are fixed
    by the caller's scope.
    """
    if engine in cache:
        return cache[engine]
    bin_name = ENGINE_TO_BINARY[engine]
    if not (BUILD / bin_name).exists():
        return None
    r = run_engine(bin_name, model, prog, max_gen, timeout=timeout)
    if r["rc"] != 0:
        cache[engine] = {"error": f"rc={r['rc']}: {r['stderr'][-200:]}"}
        return cache[engine]
    ref_path = prog.with_name(prog.stem + "_ref.txt")
    backup = prog.with_name(f"{prog.stem}_ref_{engine}.txt")
    shutil.copy(ref_path, backup)
    tokens = backup.read_text().split()
    cache[engine] = {"tokens": tokens, "wall_s": r["wall_s"]}
    return cache[engine]


def measure_pair(
    model: Path,
    prog: Path,
    engine_a: str,
    engine_b: str,
    max_gen: int,
    timeout: int,
    ref_cache: dict[str, dict],
) -> dict:
    """Measure divergence between two engines for one (model, prog)."""
    a = cached_run(ref_cache, engine_a, model, prog, max_gen, timeout)
    b = cached_run(ref_cache, engine_b, model, prog, max_gen, timeout)
    if a is None or b is None:
        return {"error": f"missing binary for {engine_a} or {engine_b}"}
    if "error" in a:
        return {"error": f"{engine_a} {a['error']}"}
    if "error" in b:
        return {"error": f"{engine_b} {b['error']}"}
    tokens_a, tokens_b = a["tokens"], b["tokens"]

    div_idx = first_divergence(tokens_a, tokens_b)
    total_compared = min(len(tokens_a), len(tokens_b))

    if div_idx < 0:
        return {
            "first_diverging_token": -1,
            "total_tokens_compared": total_compared,
            "divergence_cause": "none",
            "len_a": len(tokens_a),
            "len_b": len(tokens_b),
            "wall_a_s": a["wall_s"],
            "wall_b_s": b["wall_s"],
        }

    # Pass 2: re-run with --diag-at to capture per-(layer, head) winner.
    # max-gen capped just past the divergence index to avoid wasted work.
    diag_budget = div_idx + 16
    r_a2 = run_engine(ENGINE_TO_BINARY[engine_a], model, prog,
                       diag_budget, diag_at=div_idx, timeout=timeout)
    r_b2 = run_engine(ENGINE_TO_BINARY[engine_b], model, prog,
                       diag_budget, diag_at=div_idx, timeout=timeout)
    diag_a = parse_diag(r_a2["stderr"])
    diag_b = parse_diag(r_b2["stderr"])
    cause = attribute_cause(diag_a, diag_b)

    return {
        "first_diverging_token": div_idx,
        "total_tokens_compared": total_compared,
        "divergence_cause": cause,
        "len_a": len(tokens_a),
        "len_b": len(tokens_b),
        "wall_a_s": a["wall_s"],
        "wall_b_s": b["wall_s"],
        "tok_a": tokens_a[div_idx] if div_idx < len(tokens_a) else "<eof>",
        "tok_b": tokens_b[div_idx] if div_idx < len(tokens_b) else "<eof>",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=REPO / "models")
    ap.add_argument("--progs",  type=Path, default=REPO / "transformer_vm" / "data")
    ap.add_argument("--max-gen", type=int, default=10_000_000,
                    help="Per-engine generation cap (issue calls for 10M)")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--configs", nargs="*",
                    default=["default", "max_ffn_32", "no_reuse"])
    ap.add_argument("--progs-list", nargs="*",
                    default=["collatz", "min_cost_matching", "sudoku"])
    ap.add_argument("--pairs", nargs="*",
                    default=["naive,blas", "naive,sparse", "blas,sparse"])
    ap.add_argument("--out", type=Path,
                    default=RESULTS / "divergence_horizon.tsv")
    args = ap.parse_args()

    pairs = [tuple(p.split(",")) for p in args.pairs]
    for a, b in pairs:
        if a not in ENGINE_TO_BINARY or b not in ENGINE_TO_BINARY:
            print(f"unknown engine in pair {a},{b}", file=sys.stderr)
            return 1

    RESULTS.mkdir(exist_ok=True)
    cols = [
        "program", "engine_a", "engine_b", "model_config",
        "first_diverging_token", "total_tokens_compared", "divergence_cause",
        "len_a", "len_b", "tok_a", "tok_b",
        "wall_a_s", "wall_b_s",
    ]
    rows: list[dict] = []

    for cfg in args.configs:
        model = args.models / f"{cfg}.bin"
        if not model.exists():
            print(f"  ✗ missing model {model} — skip"); continue
        for prog_name in args.progs_list:
            prog = args.progs / f"{prog_name}.txt"
            if not prog.exists():
                print(f"  ✗ missing prog {prog} — skip"); continue
            # ref_cache is per-(prog, model): three pairs over the same
            # program share the same engine outputs. Avoids redundant runs.
            ref_cache: dict[str, dict] = {}
            for (a, b) in pairs:
                print(f"  ▶ {cfg:<11} {prog_name:<20} {a:>6}↔{b:<6}  ",
                      end="", flush=True)
                t0 = time.time()
                r = measure_pair(model, prog, a, b, args.max_gen,
                                 args.timeout, ref_cache)
                dt = time.time() - t0
                if "error" in r:
                    print(f"  ✗ {r['error']}")
                    continue
                fdt = r["first_diverging_token"]
                cause = r["divergence_cause"]
                tcomp = r["total_tokens_compared"]
                if fdt == -1:
                    print(f"  ✓ no divergence in {tcomp:,} tokens "
                          f"(wall {dt:.1f}s)")
                else:
                    print(f"  ⚠ diverge @ tok {fdt:,}/{tcomp:,}  "
                          f"cause={cause}  "
                          f"({r['tok_a']!r} vs {r['tok_b']!r}, "
                          f"wall {dt:.1f}s)")
                rows.append({
                    "program": prog_name,
                    "engine_a": a, "engine_b": b,
                    "model_config": cfg,
                    **r,
                })

    with args.out.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\n  → {args.out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
