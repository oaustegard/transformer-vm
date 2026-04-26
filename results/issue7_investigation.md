# Issue #7 — Flat-vector CHT envelope (measure first, swap if small)

**Outcome: negative result.** The two-CHT `std::multiset` layout in
`hull2d_cht.h` is kept as-is. The flat-vector swap is not landed.

## Step 1: measure (done)

Profile-build instrumentation in `transformer.cpp` emits, at the end of
each program, one stderr line per `(test, layer, head)`:

```
ENVELOPE\t<test>\t<layer>\t<head>\t<upper_size>\t<lower_size>
```

`scripts/measure_envelope_sizes.py` runs `transformer_sparse_prof` over
the 3 × 3 (config × prog) matrix at `--max-gen=20000` and writes per-pair
TSVs plus an aggregate summary. Sizes are the **upper-** and
**lower-envelope line counts** at end of generation.

### Distribution per config × program

Per-head-side counts (n = L × H per config; default and max_ffn_32 have
133 heads, no_reuse has 350):

| config       | prog              | side  | ≤10        | 11–25 | 26–50 | 51–100 | 101–500 | 501–5000 | >5000      |
|--------------|-------------------|-------|-----------:|------:|------:|-------:|--------:|---------:|-----------:|
| default      | collatz           | upper | 113 (85 %) |     3 |     1 |      1 |       0 |        0 | **15 (11 %)** |
| default      | collatz           | lower | 133 (100 %)|     0 |     0 |      0 |       0 |        0 |          0 |
| default      | min_cost_matching | upper | 116 (87 %) |     0 |     0 |      0 |       1 |        1 | **15 (11 %)** |
| default      | min_cost_matching | lower | 133 (100 %)|     0 |     0 |      0 |       0 |        0 |          0 |
| default      | sudoku            | upper | 116 (87 %) |     0 |     1 |      0 |       0 |        1 | **15 (11 %)** |
| default      | sudoku            | lower | 133 (100 %)|     0 |     0 |      0 |       0 |        0 |          0 |
| no_reuse     | collatz           | upper | 330 (94 %) |     3 |     1 |      1 |       0 |        0 | **15 (4 %)**  |
| no_reuse     | min_cost_matching | upper | 333 (95 %) |     0 |     0 |      0 |       1 |        1 | **15 (4 %)**  |
| no_reuse     | sudoku            | upper | 333 (95 %) |     0 |     1 |      0 |       0 |        1 | **15 (4 %)**  |
| max_ffn_32   | (same as default; identical histogram) ||||||||                                                                  |

Lower envelopes are tiny everywhere (100 % ≤ 10 lines, almost all = 1).
The story is entirely on the upper side.

### The bimodal pattern

The same 15 heads — the instruction-fetch / position-tracking heads in
layers 0, 2, 4 — grow exactly to `plen + max_gen` (every key joins the
upper envelope). The remaining heads stay at 1–10 lines, with 1–2
outliers in the 11–500 range.

Per-head-`upper` envelope at max_gen = 20000, default config × collatz:

```
size  count
   1     94
   2      1
   3     17
   6      4
  41      1
  71      1
24070    15   <- linear-growth heads, size = plen + max_gen
```

The 15 linear-growth heads scale linearly with the generated token
count; on full sudoku (~1 M tokens) they would reach ~1 M lines.

## Step 2: decision (skip the swap)

### Why the histogram looks like a "yes"

The issue's literal criterion was: **swap if 90 % + of heads end with
envelope < 50 lines**. Across the matrix that's clearly satisfied —
between 92 % and 98 % of heads have ≤ 10-line envelopes.

### Why the swap still loses

The criterion was implicitly designed for a unimodal distribution. The
real distribution is bimodal: a small fraction of heads do almost all
the envelope-maintenance work, and those are precisely the heads that
hurt a flat vector.

For a head where `h` grows linearly with `n` (the linear-growth
instruction-fetch heads), per-token costs are:

| op            | `std::multiset` (current)   | `std::vector<Line>` (proposed) |
|---------------|-----------------------------|---------------------------------|
| `lower_bound` | O(log h), pointer chase     | O(log h), cache-friendly        |
| insert        | O(log h) amortized          | O(h) memmove                    |
| erase         | O(log h) amortized          | O(h) memmove                    |

Total insert work over the lifetime of one such head:

- Multiset: Σ O(log h) = O(n log n)
- Vector:   Σ O(h)     = O(n²)

At `n = 1 M` (full sudoku), that's ≈ 1 M × 20 ≈ 2 × 10⁷ node-walks vs
≈ 5 × 10¹¹ element-shifts per linear-growth head — **a four- to
five-orders-of-magnitude gap on the critical heads**, before counting
the 14 others. The 89 % of heads where the vector wins (going from a
log-h multiset walk to a 1-3-element binary search) save only constant
work per insert; that fraction of total cost is already negligible.

In short: most heads are small, but most of the time isn't spent in
small heads.

### Empirical attempt

A flat-vector prototype was written (`hull2d_cht_vec.h` translating
the same `add_line`/`argmax` logic with indices instead of multiset
iterators). It compiles and runs; small workloads (≤ 6 K tokens
collatz) are even slightly faster than the multiset baseline because
memmove on tiny envelopes is cheaper than the `std::multiset`
red-black rebalancing.

But the prototype produces token streams that diverge from the
multiset baseline starting around token 4369 on collatz (and earlier
on sudoku). The bug is somewhere in the breakpoint-maintenance
sequence — translating the multiset's iterator-driven
"insert / right-sweep / left-fixup / left-walk" into index arithmetic
introduces an off-by-one that only manifests once the envelope
exceeds a few hundred lines. Debugging it properly would mean
re-deriving the maintenance invariants for the index form, which is
not justified given the analytical case above. The prototype is not
checked in.

## Out of scope

- A hybrid "vector for h < threshold, multiset above" structure was
  excluded by the issue. The histogram does not strongly demand it
  either: the small-envelope heads aren't the bottleneck, so winning
  on them doesn't move the bottom line. Switching strategies in the
  middle of a head's lifetime would also need to copy aggregated
  metadata, eating into the savings.

- Replacing the CHT algorithm itself (Li Chao, persistent CHT) — out
  of scope by issue #7 statement.

- Issue #6's upper/lower merge is also a negative result, so the
  envelope counts in this measurement are with the current two-CHT
  layout. There is no future state in which both halves merge and
  the per-head envelope sizes halve; the bimodal upper-envelope
  distribution is intrinsic to the workload.

## Files

- Instrumentation: `transformer_vm/model/transformer.cpp` (PROFILE-gated
  `ENVELOPE\t…` stderr emission per layer-head at end of each program).
- Measurement script: `scripts/measure_envelope_sizes.py`.
- Per-pair histograms: `results/envelope_sizes_<config>_<prog>.tsv`
  (one row per layer-head, columns `test layer head upper lower`).
- Combined: `results/envelope_sizes_summary.tsv` (long-format histogram
  buckets per config × prog × side).

## Reproduce

```bash
uv sync
uv run python -m transformer_vm.build --save-weights=models/default.bin
uv run python -m transformer_vm.build --save-weights=models/no_reuse.bin --no-reuse
uv run python -m transformer_vm.build --save-weights=models/max_ffn_32.bin --max-ffn=32
uv run wasm-compile --all
mkdir -p data && cp transformer_vm/data/*.txt data/
uv run wasm-reference data/collatz.txt
uv run wasm-reference data/min_cost_matching.txt
uv run wasm-reference data/sudoku.txt

ATTN=transformer_vm/attention; SRC=transformer_vm/model/transformer.cpp
COM="-std=c++17 -O3 -march=native -I $ATTN"
g++ $COM -DPROFILE_PHASES                  $SRC -o build/transformer_naive_prof
g++ $COM -DPROFILE_PHASES -DUSE_SPARSE_PROJ $SRC -o build/transformer_sparse_prof

uv run python scripts/measure_envelope_sizes.py --max-gen 20000
```
