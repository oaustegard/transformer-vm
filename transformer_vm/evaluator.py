#!/usr/bin/env python3
"""Evaluate WASM programs through the computation graph with exact arithmetic.

This runs the graph DSL directly — no transformer weights needed.
Use wasm-run for transformer inference instead.
"""

import argparse
import glob
import logging
import math
import os
import time

import numpy as np

from transformer_vm.graph.core import (
    CumSumDimension,
    LookUpDimension,
    PersistDimension,
    ReGLUDimension,
)

logger = logging.getLogger(__name__)


# ── Graph construction ────────────────────────────────────────────


def _build_default_graph():
    """Build the WASM machine graph."""
    from transformer_vm.graph.core import _all_dims, _all_lookups
    from transformer_vm.wasm.interpreter import build

    input_tokens, output_tokens = build()
    return input_tokens, output_tokens, list(_all_dims), list(_all_lookups)


_default_graph = None


def _get_default_graph():
    global _default_graph
    if _default_graph is None:
        _default_graph = _build_default_graph()
    return _default_graph


# ── Hull-based attention (optional) ─────────────────────────────────
_hull_ext = None


def _load_hull():
    global _hull_ext
    if _hull_ext is not None:
        return _hull_ext
    try:
        from torch.utils.cpp_extension import load

        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attention")
        _hull_ext = load(
            name="hull_ext",
            sources=[os.path.join(root, "hull_ext.cpp")],
            extra_cflags=["-O3", "-std=c++17"],
            extra_include_paths=[root],
            verbose=False,
        )
        return _hull_ext
    except Exception as e:
        logger.warning("Could not load hull extension: %s", e)
        return None


class HullAttention:
    """Per-lookup O(log n) attention using pybind11 convex hull."""

    def __init__(self, lookup, ext):
        self.lookup = lookup
        nv = len(lookup.value_exprs)
        self.num_value_pairs = (nv + 1) // 2
        self.nv = nv
        self.cache = ext.HullKVCache(1, self.num_value_pairs)
        self.seq = -1

    def destroy(self):
        pass

    def clear(self):
        self.cache.clear()
        self.seq = -1

    def insert_and_query(self, vals, seq):
        lu = self.lookup
        kx = lu.key_exprs_2d[0].evaluate(vals)
        ky = lu.key_exprs_2d[1].evaluate(vals)
        qx = lu.query_exprs_2d[0].evaluate(vals)
        qy = lu.query_exprs_2d[1].evaluate(vals)

        nvp = self.num_value_pairs
        raw_vals = [v.evaluate(vals) for v in lu.value_exprs]
        while len(raw_vals) < nvp * 2:
            raw_vals.append(0.0)

        keys = np.zeros((nvp, 2))
        values = np.zeros((nvp, 2))
        queries = np.zeros((nvp, 2))
        for p in range(nvp):
            keys[p, 0] = kx
            keys[p, 1] = ky
            values[p, 0] = raw_vals[p * 2]
            values[p, 1] = raw_vals[p * 2 + 1]
            queries[p, 0] = qx
            queries[p, 1] = qy

        self.seq += 1
        out = self.cache.layer_step(0, keys, queries, values, self.seq)
        out_flat = out.reshape(-1).tolist()
        return out_flat[: self.nv]


class BruteAttention:
    """Per-lookup brute-force attention cache."""

    def __init__(self, lookup):
        self.lookup = lookup
        self.entries = []  # (seq, kx, ky, [val0, val1, ...])

    def clear(self):
        self.entries.clear()

    def insert_and_query(self, vals, seq):
        lu = self.lookup
        kx = lu.key_exprs_2d[0].evaluate(vals)
        ky = lu.key_exprs_2d[1].evaluate(vals)
        raw_vals = [v.evaluate(vals) for v in lu.value_exprs]
        self.entries.append((seq, kx, ky, raw_vals))

        qx = lu.query_exprs_2d[0].evaluate(vals)
        qy = lu.query_exprs_2d[1].evaluate(vals)

        best_score = -1e300
        for _s, ekx, eky, _ev in self.entries:
            score = qx * ekx + qy * eky
            if score > best_score + 1e-9:
                best_score = score

        if lu.tie_break == "average":
            total = [0.0] * len(lu.value_exprs)
            count = 0
            for _s, ekx, eky, ev in self.entries:
                score = qx * ekx + qy * eky
                if abs(score - best_score) <= 1e-9:
                    for j in range(len(ev)):
                        total[j] += ev[j]
                    count += 1
            return [t / count for t in total] if count > 0 else total
        else:
            best_seq = -1
            best_vals = None
            for s, ekx, eky, ev in self.entries:
                score = qx * ekx + qy * eky
                if abs(score - best_score) <= 1e-9 and s > best_seq:
                    best_seq = s
                    best_vals = ev
            return list(best_vals)


# ── Runtime evaluator ────────────────────────────────────────────────


class Runtime:
    """General graph evaluator for any ProgramGraph."""

    def __init__(self, use_hull=False, program_graph=None):
        self.use_hull = use_hull
        self.hull_ext = _load_hull() if use_hull else None
        if use_hull and self.hull_ext is None:
            logger.warning("Hull extension not found, falling back to brute-force")
            self.use_hull = False

        if program_graph is not None:
            self.input_tokens = program_graph.input_tokens
            self.output_tokens = program_graph.output_tokens
            self.all_dims = program_graph.all_dims
            self.all_lookups = program_graph.all_lookups
            self._one = program_graph.one
            self._position = program_graph.position
            self._position_sq = program_graph.position_sq
            self._inv_log_pos = program_graph.inv_log_pos
        else:
            it, ot, ad, al = _get_default_graph()
            self.input_tokens = it
            self.output_tokens = ot
            self.all_dims = ad
            self.all_lookups = al
            from transformer_vm.graph.core import inv_log_pos, one, position, position_sq

            self._one = one
            self._position = position
            self._position_sq = position_sq
            self._inv_log_pos = inv_log_pos

        self.reset()

    def reset(self):
        self.pos = 0
        self.cumsum_accum = {}
        for d in self.all_dims:
            if isinstance(d, CumSumDimension):
                self.cumsum_accum[d] = 0.0

        self.attention = {}
        for lu in self.all_lookups:
            if self.use_hull:
                self.attention[lu.id] = HullAttention(lu, self.hull_ext)
            else:
                self.attention[lu.id] = BruteAttention(lu)

    def step(self, token_name):
        """Process one token. Returns dict of dimension values."""
        vals = {}

        embedding = self.input_tokens.get(token_name)
        if embedding is None:
            raise ValueError(f"Unknown token: {token_name}")
        for dim, coeff in embedding.terms.items():
            vals[dim] = coeff
        vals[self._position] = float(self.pos)
        vals[self._position_sq] = float(self.pos) ** 2
        vals[self._inv_log_pos] = 1.0 / math.log(2) - 1.0 / math.log(self.pos + 2)

        processed_lookups = {}

        for d in self.all_dims:
            if d in vals:
                continue

            if isinstance(d, CumSumDimension):
                self.cumsum_accum[d] += d.value_expr.evaluate(vals)
                vals[d] = self.cumsum_accum[d]

            elif isinstance(d, ReGLUDimension):
                a = d.a_expr.evaluate(vals)
                b = d.b_expr.evaluate(vals)
                vals[d] = a * max(0.0, b)

            elif isinstance(d, PersistDimension):
                vals[d] = d.expr.evaluate(vals)

            elif isinstance(d, LookUpDimension):
                lu = d.lookup
                if lu.id not in processed_lookups:
                    attn = self.attention[lu.id]
                    result = attn.insert_and_query(vals, self.pos)
                    processed_lookups[lu.id] = result
                vals[d] = processed_lookups[lu.id][d.value_index]

        self.pos += 1
        return vals

    def predict_next(self, vals):
        """Score all output tokens and return the argmax."""
        best_score = -1e300
        best_name = None
        for name, score_expr in self.output_tokens.items():
            score = score_expr.evaluate(vals)
            if score > best_score:
                best_score = score
                best_name = name
        return best_name

    def destroy(self):
        if self.use_hull:
            for attn in self.attention.values():
                if isinstance(attn, HullAttention):
                    attn.destroy()


def run_program(program_file, ref_file=None, use_hull=False, verbose=False):
    """Run a program and optionally compare with reference."""
    with open(program_file) as f:
        tokens = f.read().split()

    ref_tokens = None
    if ref_file and os.path.exists(ref_file):
        with open(ref_file) as f:
            ref_tokens = f.read().split()

    rt = Runtime(use_hull=use_hull)

    prog_end_idx = None
    for i, tok in enumerate(tokens):
        if tok == "}":
            prog_end_idx = i
            break

    if prog_end_idx is None:
        logger.error("No '}' found in %s", program_file)
        rt.destroy()
        return False

    # Feed program prefix (PROG_START through PROG_END)
    for i in range(prog_end_idx + 1):
        vals = rt.step(tokens[i])

    # Force-feed any input bytes after } (tokens remaining in the file)
    input_end_idx = prog_end_idx
    if prog_end_idx + 1 < len(tokens):
        for i in range(prog_end_idx + 1, len(tokens)):
            vals = rt.step(tokens[i])
            input_end_idx = i

    # Predict tokens autoregressively
    predicted = list(tokens[: input_end_idx + 1])
    output_chars = []
    draining = False
    max_steps = 50000

    for _step in range(max_steps):
        next_tok = rt.predict_next(vals)
        predicted.append(next_tok)

        if next_tok == "OUT":
            draining = True
        elif draining and next_tok != "halt":
            bv = int(next_tok, 16)
            ch = chr(bv) if 32 <= bv < 127 else "."
            output_chars.append(ch)

        if next_tok == "halt":
            break

        vals = rt.step(next_tok)

    if verbose:
        logger.info("  Tokens: %s", " ".join(predicted))

    if ref_tokens is not None:
        if predicted == ref_tokens:
            return True
        else:
            for i in range(max(len(predicted), len(ref_tokens))):
                p = predicted[i] if i < len(predicted) else "<END>"
                r = ref_tokens[i] if i < len(ref_tokens) else "<END>"
                if p != r:
                    logger.warning("  MISMATCH at position %d: predicted=%s, expected=%s", i, p, r)
                    break
            return False

    return True


# ── CLI ──────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Evaluate WASM programs through the computation graph (exact arithmetic)."
    )
    parser.add_argument("files", nargs="*", help="Program .txt files to evaluate")
    parser.add_argument(
        "--nohull",
        action="store_true",
        help="Disable O(log n) convex-hull attention (use brute-force)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print the full generated token sequence"
    )
    args = parser.parse_args()

    use_hull = not args.nohull

    files = args.files
    if not files:
        from transformer_vm._paths import DATA_DIR
        from transformer_vm.compilation.compile_wasm import ensure_data

        ensure_data()
        files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
        files = [f for f in files if not any(s in f for s in ("_ref", "_spec"))]

    any_failed = False
    for prog_file in files:
        name = os.path.basename(prog_file).replace(".txt", "")
        ref_file = prog_file.replace(".txt", "_ref.txt")
        has_ref = os.path.exists(ref_file)

        t0 = time.time()
        ok = run_program(
            prog_file, ref_file if has_ref else None, use_hull=use_hull, verbose=args.verbose
        )
        dt = time.time() - t0
        status = "PASS" if ok else "FAIL"
        logger.info("%s: %s (%.2fs)", name, status, dt)
        if not ok:
            any_failed = True

    if any_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
