#!/usr/bin/env python3
"""model_flops.py — Per-token FLOP estimates for a transformer-vm model.bin.

Issue #9: bench.py reports tok/s, which conflates *compute shape* (model
dims) with *engine shape* (sparse vs dense, hard-attn vs softmax). FLOPs
gives the substrate-fair axis. This module:

  1. Introspects model.bin (header + per-matrix nnz counts).
  2. Returns per-token FLOP estimates split by phase
     (qkv / attn / out / ffn / head) under two interpretations:
       - theoretical: dense FLOPs (engine-agnostic, depends only on dims)
       - effective:   engine-aware (sparse skips zeros, hard-attn ~log env)

The numbers are deliberately model-from-shape: no perf counters, no
runtime instrumentation. Portable, comparable across machines.

Usage as a CLI:
    uv run python scripts/model_flops.py path/to/model.bin --engine sparse
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ── Constants ──────────────────────────────────────────────────────────

# Per-envelope-op cost in the CHT insert/query path. Each level of the
# multiset walk does a 2D dot, a comparison, and a few arithmetic ops on
# slope/intercept — call it ~8 FLOPs. Coarse but defensible: attention
# is FLOP-cheap relative to projections regardless of the constant.
ATTN_FLOPS_PER_LEVEL = 8.0

# Position-encoding mutations to x: 3 adds + a log + a divide. Negligible
# vs even the smallest projection, included for completeness.
EMBED_FLOPS = 6.0


# ── Shape introspection ───────────────────────────────────────────────

@dataclass
class ModelShape:
    V: int
    D: int
    L: int
    H: int
    F: int
    stop: int

    @property
    def per_layer_dense(self) -> int:
        D, F = self.D, self.F
        return 3 * D * D + D * D + 2 * F * D + D * F


@dataclass
class LayerNNZ:
    qkv: int
    out: int
    fi: int
    fo: int

    @property
    def total(self) -> int:
        return self.qkv + self.out + self.fi + self.fo


@dataclass
class ModelStats:
    """Shape + per-matrix nonzero counts for FLOP modeling."""
    shape: ModelShape
    head_nnz: int
    layers: list[LayerNNZ] = field(default_factory=list)

    @property
    def total_proj_nnz(self) -> int:
        return sum(l.total for l in self.layers)

    @property
    def total_proj_dense(self) -> int:
        return self.shape.L * self.shape.per_layer_dense


def _read_header(f) -> tuple[ModelShape, list[str]]:
    raw = f.read(24)
    V, D, L, H, F, stop = struct.unpack("<6i", raw)
    names = []
    for _ in range(V):
        (n,) = struct.unpack("<I", f.read(4))
        names.append(f.read(n).decode("utf-8"))
    return ModelShape(V=V, D=D, L=L, H=H, F=F, stop=stop), names


def load_shape(path: str | Path) -> ModelShape:
    """Cheap: header only."""
    with open(path, "rb") as f:
        shape, _ = _read_header(f)
    return shape


def load_stats(path: str | Path) -> ModelStats:
    """Read full weights, count nonzeros per matrix."""
    with open(path, "rb") as f:
        shape, _ = _read_header(f)
        D, L, F, V = shape.D, shape.L, shape.F, shape.V

        emb = np.frombuffer(f.read(V * D * 8), dtype=np.float64)
        # Embedding nnz isn't used for projection-cost modeling, just consumed.
        _ = emb

        layers: list[LayerNNZ] = []
        for _i in range(L):
            qkv = np.frombuffer(f.read(3 * D * D * 8), dtype=np.float64)
            out = np.frombuffer(f.read(D * D * 8),     dtype=np.float64)
            fi  = np.frombuffer(f.read(2 * F * D * 8), dtype=np.float64)
            fo  = np.frombuffer(f.read(D * F * 8),     dtype=np.float64)
            layers.append(LayerNNZ(
                qkv=int(np.count_nonzero(qkv)),
                out=int(np.count_nonzero(out)),
                fi=int(np.count_nonzero(fi)),
                fo=int(np.count_nonzero(fo)),
            ))

        head = np.frombuffer(f.read(V * D * 8), dtype=np.float64)
        head_nnz = int(np.count_nonzero(head))

    return ModelStats(shape=shape, head_nnz=head_nnz, layers=layers)


# ── FLOP model ─────────────────────────────────────────────────────────

def _attn_flops_per_layer(H: int, envelope: float) -> float:
    """Per-token attention cost: H heads × ~log2(envelope) levels × per-level work.

    The CHT does an insert + a query each step. Both walk O(log h) levels of
    the multiset. We charge 2 walks × log2(env) × ATTN_FLOPS_PER_LEVEL × H.
    """
    if envelope < 2:
        return float(H) * 2.0 * ATTN_FLOPS_PER_LEVEL
    return float(H) * 2.0 * math.log2(envelope) * ATTN_FLOPS_PER_LEVEL


def per_token_flops(
    stats: ModelStats,
    engine: str,
    n_tok: int | None = None,
    prompt_len: int = 0,
    avg_envelope: float | None = None,
) -> dict:
    """Per-token FLOP breakdown for one engine path.

    Args:
        engine: "naive" | "blas" | "sparse". naive and blas are dense;
                sparse uses CSR matvec with nnz-aware FLOP count.
        n_tok:  total tokens in the run (prefix + generated). Used to
                amortize the head, which fires once per generated token.
                If None, head is counted as if it fires every position.
        prompt_len: tokens consumed without generation. Excluded from
                head amortization.
        avg_envelope: typical CHT envelope size during the run. If None,
                we use n_tok / 4 as a coarse default (real measurements
                from issue #7 show 85-95% of layer-heads stay tiny, but
                a handful grow linearly — this average is a wash).

    Returns a dict with keys:
        embed, qkv, attn, out, ffn, head, total
    Each value is per-token FLOPs (averaged across the run).
    """
    if engine not in ("naive", "blas", "sparse"):
        raise ValueError(f"unknown engine: {engine!r}")
    s = stats.shape
    D, L, H, F, V = s.D, s.L, s.H, s.F, s.V

    if avg_envelope is None:
        if n_tok and n_tok > 4:
            avg_envelope = n_tok / 4.0
        else:
            avg_envelope = 1.0

    # Per-layer projection FLOPs (matvec: 2 * rows * cols).
    if engine == "sparse":
        per_layer_proj = sum(2 * l.total for l in stats.layers) / max(L, 1)
        # FFN GLU multiply: F muls per layer (negligible but real).
        per_layer_ffn_extra = float(F)
        # Split by phase using each layer's nnz mix; take the average.
        qkv_pt = 2.0 * sum(l.qkv for l in stats.layers) / max(L, 1)
        out_pt = 2.0 * sum(l.out for l in stats.layers) / max(L, 1)
        fi_pt  = 2.0 * sum(l.fi  for l in stats.layers) / max(L, 1)
        fo_pt  = 2.0 * sum(l.fo  for l in stats.layers) / max(L, 1)
        ffn_pt = fi_pt + fo_pt + per_layer_ffn_extra
    else:
        qkv_pt = 2.0 * 3 * D * D
        out_pt = 2.0 * D * D
        fi_pt  = 2.0 * 2 * F * D
        fo_pt  = 2.0 * D * F
        ffn_pt = fi_pt + fo_pt + float(F)

    qkv_flops = L * qkv_pt
    out_flops = L * out_pt
    ffn_flops = L * ffn_pt
    attn_flops = L * _attn_flops_per_layer(H, avg_envelope)

    # Head: argmax over V·D. The C++ engine always uses the sparse CSR
    # path for the head (transformer.cpp:419), regardless of build flag,
    # so effective head = 2 * head_nnz universally. We separate the
    # naming so callers can request the dense interpretation if desired.
    head_dense = 2.0 * V * D
    head_eff   = 2.0 * stats.head_nnz
    head_flops = head_dense if engine in ("naive", "blas") and stats.head_nnz == V * D else head_eff

    if n_tok and n_tok > 0:
        head_share = max(0, n_tok - prompt_len) / n_tok
        head_flops *= head_share

    embed_flops = EMBED_FLOPS

    total = embed_flops + qkv_flops + attn_flops + out_flops + ffn_flops + head_flops
    return {
        "embed": embed_flops,
        "qkv":   qkv_flops,
        "attn":  attn_flops,
        "out":   out_flops,
        "ffn":   ffn_flops,
        "head":  head_flops,
        "total": total,
    }


def per_token_theoretical(
    stats: ModelStats,
    n_tok: int | None = None,
    prompt_len: int = 0,
    avg_envelope: float | None = None,
) -> dict:
    """Engine-agnostic dense FLOPs — the 'shape' axis only.

    Equivalent to per_token_flops(stats, 'naive', ...) but with the head
    forced dense (V·D) regardless of head_sp. This is the number to
    hold steady when comparing engines on the same model.
    """
    s = stats.shape
    D, L, H, F, V = s.D, s.L, s.H, s.F, s.V

    if avg_envelope is None:
        avg_envelope = (n_tok / 4.0) if (n_tok and n_tok > 4) else 1.0

    qkv_flops = L * 2.0 * 3 * D * D
    out_flops = L * 2.0 * D * D
    ffn_flops = L * (2.0 * 2 * F * D + 2.0 * D * F + float(F))
    attn_flops = L * _attn_flops_per_layer(H, avg_envelope)
    head_flops = 2.0 * V * D
    if n_tok and n_tok > 0:
        head_flops *= max(0, n_tok - prompt_len) / n_tok

    total = EMBED_FLOPS + qkv_flops + attn_flops + out_flops + ffn_flops + head_flops
    return {
        "embed": EMBED_FLOPS,
        "qkv":   qkv_flops,
        "attn":  attn_flops,
        "out":   out_flops,
        "ffn":   ffn_flops,
        "head":  head_flops,
        "total": total,
    }


# ── CLI ────────────────────────────────────────────────────────────────

def _fmt_g(x: float) -> str:
    return f"{x / 1e9:>8.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path, help="Path to model.bin")
    ap.add_argument("--engine", default="all",
                    choices=["all", "naive", "blas", "sparse"])
    ap.add_argument("--n-tok", type=int, default=None,
                    help="Total tokens (for head amortization)")
    ap.add_argument("--prompt-len", type=int, default=0)
    ap.add_argument("--avg-envelope", type=float, default=None,
                    help="Average CHT envelope size (default: n_tok/4)")
    args = ap.parse_args()

    stats = load_stats(args.model)
    s = stats.shape
    print(f"\nmodel: {args.model}")
    print(f"  V={s.V} D={s.D} L={s.L} H={s.H} F={s.F}")
    print(f"  per-layer dense: {s.per_layer_dense:,}")
    print(f"  proj nnz: {stats.total_proj_nnz:,} / {stats.total_proj_dense:,} "
          f"({100*(1 - stats.total_proj_nnz/stats.total_proj_dense):.1f}% sparse)")
    print(f"  head nnz: {stats.head_nnz:,} / {s.V*s.D:,} "
          f"({100*(1 - stats.head_nnz/(s.V*s.D)):.1f}% sparse)")

    engines = ["naive", "blas", "sparse"] if args.engine == "all" else [args.engine]
    th = per_token_theoretical(stats, n_tok=args.n_tok,
                                prompt_len=args.prompt_len,
                                avg_envelope=args.avg_envelope)
    print(f"\nper-token GFLOPs (theoretical, dense-equivalent):")
    print(f"  embed={_fmt_g(th['embed'])}  qkv={_fmt_g(th['qkv'])}  "
          f"attn={_fmt_g(th['attn'])}  out={_fmt_g(th['out'])}  "
          f"ffn={_fmt_g(th['ffn'])}  head={_fmt_g(th['head'])}  "
          f"total={_fmt_g(th['total'])}")

    for eng in engines:
        ef = per_token_flops(stats, eng, n_tok=args.n_tok,
                              prompt_len=args.prompt_len,
                              avg_envelope=args.avg_envelope)
        speedup = th["total"] / ef["total"] if ef["total"] > 0 else float("inf")
        print(f"\nper-token GFLOPs (effective, engine={eng}):")
        print(f"  embed={_fmt_g(ef['embed'])}  qkv={_fmt_g(ef['qkv'])}  "
              f"attn={_fmt_g(ef['attn'])}  out={_fmt_g(ef['out'])}  "
              f"ffn={_fmt_g(ef['ffn'])}  head={_fmt_g(ef['head'])}  "
              f"total={_fmt_g(ef['total'])}")
        print(f"  → {speedup:.2f}× less work than dense")

    return 0


if __name__ == "__main__":
    sys.exit(main())
