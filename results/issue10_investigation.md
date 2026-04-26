# Issue #10 — Decision-boundary divergence horizon

**Status: measurement complete for `default` and `max_ffn_32` configs.
`no_reuse` partial (collatz + mcm done; sudoku skipped — see "Why
no_reuse sudoku is missing" below). Step 2 (analytical prediction)
deferred.**

## Setup

The phenomenon is deterministic FP rounding, not non-determinism. BLAS's
vectorized FMA / blocked accumulation orders the dot-product summation
differently from the naive scalar loop, so identical mathematical inputs
produce results that differ in the last few mantissa bits. For a single
matvec this is invisible; over millions of forward passes it compounds
through the residual stream until either:

1. The internal hard-attention argmax in some `(layer, head)` flips —
   call this **`attn_argmax`** — selecting a different historical key.
   The attention output then differs structurally, not just in low bits,
   and the divergence cascades.
2. The final head-logit argmax flips with no upstream attn flip — call
   this **`head_argmax`**. The two engines computed the same attention
   selections layer-by-layer but the head's top-2 logits were close
   enough that the cumulative residual rounding tipped which token wins.

Sparse vs naive shares scalar summation order (sparse iterates only the
non-zero columns, in the same column order naive's dense loop visits, and
`s += 0.0 * x[j]` is bitwise-exact), so they are expected to be
byte-identical for the full budget. The interesting pairs are the ones
that cross summation strategies: `naive↔blas` and `blas↔sparse`.

## Tool: `--diag-at=N`

Added to `transformer.cpp`. When set, the engine emits one stderr line
just before producing token `ids[N]` and exits:

```
DIAG <test> <pos> <head_best> <head_second> <best_score> <second_score> <kx_l0_h0> ... <kx_lL_hH>
```

The per-`(layer, head)` `kx` is the slope of the winning hard-attention
line in that head's CHT envelope at this pos (or, for the qy=0 boundary
case, the signed key-x bound). `HardAttentionHead::query` and
`BruteAttentionHead::query` take an optional `double* best_kx_out` so the
hot path keeps zero overhead when the flag is not set. The engine
returns immediately after emitting DIAG, so cause-attribution re-runs
cost only `O(div_idx)` forward passes — not `O(max_gen)`.

`scripts/divergence_horizon.py` orchestrates a two-pass measurement:

1. **Pass 1**: run engine A and engine B with `--regen --max-gen=B`.
   Snapshot each engine's `_ref.txt`, then byte-compare the token streams
   to find the first index `N` where they differ. Engine outputs are
   cached per `(model, prog)` so the three pairs (A↔B, A↔C, B↔C) only
   trigger three engine runs, not six.
2. **Pass 2**: if `N >= 0`, re-run both engines with `--diag-at=N`.
   Parse each engine's DIAG line. Bitwise-compare the per-head `kx`
   vector — any difference means `attn_argmax` flipped first; if all
   `kx` match but `head_best` differs, attribute to `head_argmax`. If
   neither differs the cause is `unknown` (shouldn't happen for genuine
   divergence).

`%.17g` formatting on the `kx` values guarantees that bitwise-identical
doubles round-trip identically — the comparison is exact.

## Results

`results/divergence_horizon.tsv`. Headlines:

### Sudoku diverges at exactly token 1,671,695 — independent of FFN width

| config       | naive↔blas             | naive↔sparse | blas↔sparse              |
|--------------|------------------------|--------------|--------------------------|
| `default`    | div @ 1,671,695, attn  | none in 2.04M| div @ 1,671,695, attn    |
| `max_ffn_32` | div @ 1,671,695, attn  | none in 2.04M| div @ 1,671,695, attn    |

`tok_a / tok_b` at the divergence boundary is `'03' / 'df'` for naive↔blas
and the inverse `'df' / '03'` for blas↔sparse — confirming sparse
behaves identically to naive (it's blas that flipped the argmax).

The divergence position is identical across `default` and `max_ffn_32`
because both share the same `D=38, H=19` attention parameters; only `F`
(FFN inner dim) differs (`44` vs `32`). The sudoku divergence is
attention-driven (`cause=attn_argmax`), and FFN width affects only the
residual contribution, not the Q/K matvec rounding that decides the
attn argmax flip. So the horizon is invariant to `F` here.

### Sparse↔naive byte-identical for full 2M-token budget

Validates the methodology: sparse and naive use the same summation
order on identical operands (sparse skips zero columns; the surviving
column order is identical to naive's dense loop, and `s += 0.0 * x[j]`
is exact). On `default` and `max_ffn_32` this held for the full
~2,037,473 tokens of sudoku — `first_diverging_token = -1`.

### Short programs don't reach the horizon

`collatz` halts at ~6.1k tokens, `min_cost_matching` at ~40.2k. Both
finish well before the FP-rounding horizon under any tested config,
so all three engine pairs match byte-for-byte to halt. The horizon
phenomenon is invisible to programs that complete in <100k tokens —
which means it's invisible to current `bench.py` and `bench_real.py`
defaults.

### Why `no_reuse` sudoku is missing

`no_reuse` builds a much larger model (`D=98, H=49` vs default's
`D=38, H=19`), so the sudoku 2M-token sweep is roughly 4–5× slower
per token. A full naive↔blas + blas↔sparse sweep with cause-attribution
re-runs runs >60 minutes on this machine. Cut for now; the
infrastructure is in place — re-run with:

```
uv run python scripts/divergence_horizon.py --configs no_reuse --progs-list sudoku
```

The interesting question for `no_reuse` specifically is whether the
larger `D` shifts the horizon (more entries in each `D × D` dot product
→ more rounding per matvec → expected: shorter horizon). That's a
data-point worth getting if and when somebody runs it; it doesn't change
the headline finding.

## Implications (per the issue)

- **Bounds the correctness window per engine pair**. For long-running
  programs (`>1M` tokens) on the default-shape model, naive↔blas
  decouples at a measurable horizon (~1.67M for sudoku here). For runs
  beyond that, fall back to a scalar-order-deterministic engine
  (naive or sparse) if bit-exactness with naive is required.
- **Identifies whether `--max-gen` defaults are too short to catch this
  in CI.** They are. `bench.py` runs ~500 generated tokens; `bench_real`
  defaults to 20k. Both finish three orders of magnitude before the
  observed horizon. Engines silently disagreeing past token N is
  invisible to CI today.
- **Cause is `attn_argmax`, not `head_argmax`**, on all sudoku
  divergences observed. The attention layer is the lever — a
  quantize-decision-heads pass would only need to harden the heads
  whose argmax becomes unstable at this scale, not the head logits.
  (The diag stream tells you which heads are sensitive: bitwise-compare
  the `kx_lL_hH` columns at the divergence boundary.)

## Step 2 (deferred)

Predicting the horizon analytically: ULP per dot product
(`D × machine epsilon` × condition factor) × decisions per token (1 head
argmax + L·H attn argmax) × the decision-margin distribution at each
boundary → expected horizon. Worth doing only if a measured horizon
is short enough to bite real workloads. For sudoku at default ~10⁶
tokens vs typical real runs of ~10⁴ tokens, current mitigation cost
isn't justified.

## Non-goals (re-stated)

- **Not** fixing the divergence. Determinism between engines is doable
  (force scalar order in BLAS, plus a scalar-summation override) but
  it would discard the BLAS speedup. The point here is to characterize,
  not mitigate.
- **Not** a general theory of FP error in attention; we just need
  per-config horizon numbers.

## Note on fixtures

The issue lists `countdown` as a fourth program. It does not exist in
the current repo (`transformer_vm/examples/` has collatz, addition,
fibonacci, hello, lowering_test, min_cost_matching, sudoku). The sweep
ran on the three programs that are in the repo and are non-trivial in
length: collatz, min_cost_matching, sudoku. If `countdown` is added
later, re-run with `--progs-list collatz min_cost_matching sudoku
countdown`.
