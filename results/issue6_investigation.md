# Issue #6 — Merge upper + lower CHTs in HardAttentionHead

**Outcome: negative result.** No commit lands on `main`. The two-CHT
layout in `hull2d_cht.h` is kept as-is.

## Goal

PR #5 confirmed `attn` is ~70% of token time on the sparse engine.
`HardAttentionHead` keeps two `_HullCHT` envelopes (upper and lower);
each `insert(key, val, seq)` call runs both `add_line` walks, even
though queries only touch one envelope per token. Issue #6 proposed
folding the two envelopes into one to halve the insert cost — with a
1.2-1.4× tok/s ceiling per Amdahl.

## Approaches tried

### A. Single CHT, derived lower from upper (issue's option (c))

Discarded on inspection. The min of `kx*m + ky` over an arbitrary
point set is *not* generally on the upper convex hull. Counter-example:
points (0,0), (1,-10), (2,0). At m=-1, the min is point (1,-10) — but
(1,-10) is dominated in the upper envelope and gets pruned. Querying
the upper envelope alone returns (2,0) instead. Wrong answer.
Same logic rules out option (b) literally as written ("same set of
lines under sign-flipped slope ordering"): the surviving lines in the
upper envelope are not the surviving lines in the lower envelope.

### B. Shared per-(kx, ky) PointAggregate with dedup map

Implemented: `HardAttentionHead` owns a `deque<PointAggregate>` and an
`unordered_map<pair<double,double>, PointAggregate*>`. New keys
allocate one aggregate, and both `HullHalf`s' Lines hold a raw pointer
to it. Repeat (kx, ky) inserts hit the map and skip both envelope
walks entirely.

Correctness: byte-identical to brute-force on collatz and
min_cost_matching across all three engines.

Performance (real model, default config, this machine):

| Engine                | collatz tok/s | mcm tok/s |
|-----------------------|--------------:|----------:|
| baseline              |        47,950 |    52,321 |
| shared + dedup map    |    **30,225** | **30,587** |
| shared, no dedup map  |        42,078 |       n/a |

Phase breakdown (sparse engine, collatz):

| Variant            | attn share | attn time |
|--------------------|-----------:|----------:|
| baseline           |    68.7%   |    0.40s  |
| shared + dedup map |    83.1%   |    0.92s  |

The dedup map's hash-and-emplace per insert costs more than the
envelope walks it saves, because the workload is essentially
collision-free — QKV projections produce floats that almost never
match an earlier (kx, ky). The unordered_map allocates and probes on
every insert and gives back nothing.

### C. Shared PointAggregate without the dedup map

Drops the map; each insert still allocates one aggregate, both
envelopes' Lines hold the same pointer, `add_line` re-acquires its
in-envelope merge logic for same-(m,b) collisions. Eliminates the
hash-map cost but keeps the pointer-indirection cost on every
metadata access — including the tied-merge walk inside `query`.

Result: 12% slower than baseline on collatz sparse (42k vs 48k tok/s).
The indirection is a real cache hit — `Line::meta` was inlined; now
each meta read is an extra dereference into an unrelated heap node.

## Why option (b) is fundamentally not free

The two envelopes are over different surviving point sets, which is a
geometric fact about the data, not a representation choice. Sharing
storage (aggregates, lines, anything) only reduces metadata write
cost, never the envelope-maintenance work, because both envelopes
have to do their own walk regardless. And on this workload the
envelope walks are already cheap relative to the per-Line bookkeeping
overhead the sharing adds.

The only paths that could plausibly win:

- A genuine dynamic 2D convex hull (Overmars-van Leeuwen,
  Brodal-Jacob) folding both halves into one structure with a single
  O(log²n) maintenance pass. Big rewrite, complex correctness story.
- A workload where (kx, ky) collisions are common (e.g. heavy
  quantization or sparse keys with many exact zeros). The dedup
  variant would help there. Not the case for this transformer.

## Files

The two failed variants are kept as reference branches in the commit
history of this PR (`attn: share PointAggregate...` and the no-dedup
follow-up); the final state of the branch reverts to baseline.

Reproduce with:

```bash
uv run python -m transformer_vm.build --save-weights=data/model.bin
uv run wasm-compile transformer_vm/examples/collatz.c --args 7 -o data/collatz
uv run wasm-compile transformer_vm/examples/min_cost_matching.c \
    --args "3 9 2 7 6 4 3 5 8 1" -o data/mcm
uv run wasm-reference data/collatz.txt
uv run wasm-reference data/mcm.txt

# build engines
ATTN=transformer_vm/attention
SRC=transformer_vm/model/transformer.cpp
COM="-std=c++17 -O3 -march=native -I $ATTN"
g++ $COM                    $SRC -o build/transformer_naive
g++ $COM -DUSE_OPENBLAS     $SRC -o build/transformer_blas -lopenblas
g++ $COM -DUSE_SPARSE_PROJ  $SRC -o build/transformer_sparse

uv run python scripts/bench_real.py \
    --model data/model.bin --prog data/collatz.txt --label X --repeats 3
uv run python scripts/bench_real.py \
    --model data/model.bin --prog data/mcm.txt --label X --repeats 3
```
