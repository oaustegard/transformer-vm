# Issue #9 — FLOPs-normalized throughput, applied

The metric landed in `scripts/model_flops.py` + `scripts/bench.py` +
`scripts/bench_real.py`. This note records what the metric actually
told us when run.

**Outcome.** The "18× sparse-vs-naive" headline from synthetic random
sparsity does **not** transfer to real analytical models. On real
models the speedup is **2.4×**. The compute-density story is the
same in both regimes — sparse wins by skipping work, not by being
hardware-efficient — but the magnitude of "less work" is much smaller
when the model is small.

## Method recap

Two numbers per (model, engine):

- `gflops_effective` — analytical per-token FLOPs the engine actually
  performs, computed from `model.bin` shape + `numpy.count_nonzero`
  on each weight matrix. Engine-aware: sparse uses `2 × nnz`, dense
  uses `2 × rows × cols`. Constant for a given (model, engine).
- `effective_gflops_per_s = gflops_effective × tok_s` — the rate at
  which the engine produces useful FLOPs. The substrate-fair axis:
  what fraction of the hardware's compute capacity the engine
  actually uses, with the engine-shape effect (sparsity skip)
  divided out.

If two engines achieve the same `effective_gflops_per_s`, they're
equally efficient at hardware utilization; any tok/s difference
between them is "less work to do," not "smarter execution."

## Synthetic sparsity sweep

`scripts/bench.py`, V=128 D=128 L=8 H=4 F=256, single-thread BLAS,
results in `results/sweep.tsv`:

```
sparsity    naive     BLAS    sparse    naive GF/s   BLAS GF/s   sparse GF/s
   50%      1,269    2,183     1,702        3.35        5.77         2.26
   75%      1,305    2,116     2,796        3.44        5.57         1.87
   90%      1,159    2,131     6,403        3.05        5.60         1.73
   95%      1,354    2,081     9,916        3.56        5.47         1.37
   99%      1,351    2,146    24,822        3.55        5.64         0.79
```

Naive and BLAS each pin to a flat hardware ceiling (~3.5 / ~5.6 GF/s)
across all sparsities — neither knows about zeros, both multiply
through them. Sparse's `effective_gflops_per_s` **falls monotonically
with sparsity** (2.3 → 0.8 GF/s), reaching ~14% of BLAS at 99% sparse.
The CSR gather pattern (`x[col[k]]`) breaks stride prediction, and
that tax compounds as nnz drops.

The 1.34→18.4× tok/s win is honest accounting. Sparse runs at lower
hardware density but does so much less work it still wins overall.

## Real model

`scripts/bench_real.py`, default analytical model
(V=915 D=38 L=7 H=19 F=47, 98.6 % projection sparse, 84.6 % head
sparse), single-thread BLAS, results in
`results/real_model_default_*.tsv`:

```
program              naive     BLAS    sparse    naive GF/s   BLAS GF/s   sparse GF/s   sp/naive
collatz(7)          20,941   34,221    49,594        3.89        6.36         1.60      2.37×
min_cost_matching   21,415   34,574    51,290        4.01        6.48         1.73      2.40×
```

Two programs, virtually identical numbers. Sparse beats naive by
**2.4×** on tok/s, but its hardware utilization (1.6–1.7 GF/s) is
**~25 % of BLAS** — the same penalty regime as synthetic.

## Why 18× → 2.4×

Per-token theoretical work is **12× smaller** on the real model:

| | synthetic D=128 L=8 F=256 | real D=38 L=7 F=47 |
|---|---:|---:|
| per-token GFLOPs theoretical | 0.0027 | 0.000218 |
| per-token GFLOPs sparse      | 0.000032 (99% sp) | 0.000032 |
| theoretical / sparse ratio   | 82× | 7× |

The per-layer projection cost scales with `D²` and `D × F`. Real
model has D=38, F=47 — projections are tiny in absolute terms.
Meanwhile the per-token *fixed costs* (CHT walks, head argmax over
V=915, position encoding, layer dispatch) don't shrink with the
model. So the slice of work sparsity can optimize is a much smaller
fraction of the total.

Decomposition of the real-model 2.4× win:

- **Less work**: 7× fewer FLOPs per token (98.6% projection sparse +
  84.6% head sparse).
- **Hardware penalty**: sparse runs at 0.41× of naive's GF/s
  (1.60 / 3.89).
- **Net**: 7 × 0.41 ≈ 2.9× theoretical, ~2.4× observed (gap from
  attention/embed overhead amortization).

## What this changes about the engine story

1. **Sparse is real but bounded** on real models. 2.4× is useful;
   18× was a synthetic artifact. Anyone quoting the synthetic number
   in a writeup is overclaiming.
2. **BLAS is the actual ceiling.** It hits 6.4 GF/s on the real
   model, naive hits 4.0 GF/s, sparse hits 1.7 GF/s. BLAS does **4×
   more useful FLOPs/s than sparse**, but loses on tok/s because it
   does 7× more work. A representation that combined "skip zeros"
   with BLAS-quality SIMD — blocked CSR, hybrid dense/sparse — could
   plausibly stack: ~7× from skipping, ~2-4× from utilization. On
   this model that's a 14-28× ceiling over naive, vs 2.4× today.
3. **Sparse is a misnomer at low sparsity.** From the synthetic
   sweep: at 50 % sparsity, BLAS beats sparse on tok/s (2,183 vs
   1,702). The CSR engine only pays off above ~75 % sparse for this
   D. Default-engine selection should be sparsity-aware.
4. **The metric did its job.** Without `gflops_effective`, the 2.4×
   real-model speedup looks like "a bit underwhelming" against the
   18× synthetic. With it, the explanation falls out: same
   compute-density story (sparse < naive < BLAS), just less
   projection work to amortize the indirection over.

## Reproducing

```bash
make all                              # naive + sparse + naive_prof + sparse_prof
make blas                             # blas + blas_prof (needs libopenblas-dev)

uv sync
uv run python -m transformer_vm.build --milp --save-weights=models/default.bin
uv run wasm-compile transformer_vm/examples/collatz.c --args 7

uv run python scripts/bench.py --sparsities 0.50,0.75,0.90,0.95,0.99
uv run python scripts/bench_real.py \
    --model models/default.bin \
    --prog transformer_vm/data/collatz.txt \
    --label default_collatz
```

Numbers will vary with CPU; ratios should be stable.
