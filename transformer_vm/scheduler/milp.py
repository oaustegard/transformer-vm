"""MILP scheduler: minimize d_model for delayed-reuse erase scheduling.

Assigns each gate (LookUp, ReGLU, Persist) to a 4-phase layer:
  phase 0: Attention (LookUp)
  phase 1: Persist1
  phase 2: FFN (ReGLU)
  phase 3: Persist2

Minimizes d_model = 2 * D_half where D_half >= max over all boundaries of:
  - ceil(effective_width / 2), counting dims with birth<=c AND death>=c-1
    AND needs_slot (excludes internal lookup/reglu dims consumed same half-layer)
  - n_lu_heads + ceil((dying + passthrough) / 2) per attention layer
"""

import argparse  # noqa: I001
import heapq
import logging
from collections import defaultdict

import numpy as np
from pulp import (
    LpBinary,
    LpInteger,
    LpMinimize,
    LpProblem,
    LpVariable,
    lpSum,
    value,
)

from transformer_vm.graph.core import (
    Expression,
    InputDimension,
    LookUp,
    LookUpDimension,
    PersistDimension,
    ReGLUDimension,
    _all_dims,
    _all_lookups,
    inv_log_pos as _default_inv_log_pos,
    position as _default_position,
    position_sq as _default_position_sq,
)

logger = logging.getLogger(__name__)


def _build_graph(all_dims=None, all_lookups=None, ilp=None):
    """Build dependency graph from dim/lookup lists.

    Args:
        all_dims: list of Dimension objects (defaults to module-global _all_dims)
        all_lookups: list of LookUp objects (defaults to module-global _all_lookups)
        ilp: the inv_log_pos dimension (defaults to module-global inv_log_pos)
    """
    if all_dims is None:
        all_dims = _all_dims
    if all_lookups is None:
        all_lookups = _all_lookups
    if ilp is None:
        ilp = _default_inv_log_pos
    inputs = [d for d in all_dims if isinstance(d, InputDimension)]
    reglus = [d for d in all_dims if isinstance(d, ReGLUDimension)]
    persists = [d for d in all_dims if isinstance(d, PersistDimension)]
    lookups = list(all_lookups)
    ops = reglus + persists + lookups

    produced = {}
    for r in reglus:
        produced[r] = {r}
    for p in persists:
        produced[p] = {p}
    for lu in lookups:
        produced[lu] = set(lu.dims)

    def _edeps(expr):
        return set(expr.terms.keys()) if isinstance(expr, Expression) else set()

    deps_cache = {}
    for r in reglus:
        deps_cache[r] = _edeps(r.a_expr) | _edeps(r.b_expr)
    for p in persists:
        deps_cache[p] = _edeps(p.expr)
    for lu in lookups:
        d = set()
        for expr in lu.query_exprs_2d + lu.key_exprs_2d + lu.value_exprs:
            d |= _edeps(expr)
        d.add(ilp)
        deps_cache[lu] = d

    dim_to_op = {}
    for op in ops:
        for d in produced[op]:
            dim_to_op[d] = op

    op_deps = defaultdict(set)
    children = defaultdict(set)
    consumers = defaultdict(set)
    for op in ops:
        for dim in deps_cache[op]:
            consumers[dim].add(op)
            if dim in dim_to_op and dim_to_op[dim] != op:
                pred = dim_to_op[dim]
                op_deps[op].add(pred)
                children[pred].add(op)

    avg_lookups = {lu for lu in lookups if lu.tie_break == "average"}
    tight_to = defaultdict(set)
    for op in reglus + persists:
        for dim in deps_cache[op]:
            if isinstance(dim, LookUpDimension) and dim in dim_to_op:
                lu = dim_to_op[dim]
                if lu in avg_lookups:
                    tight_to[op].add(lu)

    return dict(
        ops=ops,
        reglus=reglus,
        persists=persists,
        lookups=lookups,
        inputs=inputs,
        produced=produced,
        deps_cache=deps_cache,
        dim_to_op=dim_to_op,
        op_deps=op_deps,
        children=children,
        consumers=consumers,
        tight_to=dict(tight_to),
    )


def _min_layers(ops, op_deps):
    """Critical path length with phase parity (ASAP without tight constraints)."""
    phase, remaining = {}, set(ops)
    while remaining:
        progress = False
        for op in list(remaining):
            if not all(p in phase for p in op_deps[op]):
                continue
            lo = max((phase[p] for p in op_deps[op]), default=-1) + 1
            if isinstance(op, LookUp):
                lo += (-lo) % 4
            elif isinstance(op, ReGLUDimension):
                lo += (2 - lo % 4 + 4) % 4
            else:
                lo += 0 if lo % 2 == 1 else 1
            phase[op] = lo
            remaining.discard(op)
            progress = True
        assert progress, "Cycle in dependencies"
    return max(phase.values()) // 4 + 1


def _all_result_dims(graph):
    """All dimensions in the graph (inputs + produced)."""
    dims = list(graph["inputs"])
    dim_set = set(dims)
    for op in graph["ops"]:
        for d in graph["produced"][op]:
            if d not in dim_set:
                dim_set.add(d)
                dims.append(d)
    return dims


def milp_schedule(
    input_tokens, output_tokens, max_layers=None, max_ffn=None, log=None, program_graph=None
):
    """Optimal schedule minimizing dependency width at persist boundaries.

    Args:
        max_layers: maximum number of transformer layers.
        max_ffn: maximum FFN neurons per layer.
        log: logging callable (defaults to logger.info).
        program_graph: optional ProgramGraph; if provided, uses its dims/lookups
            and positional dimensions instead of module-level globals.
    """
    _log = log or print

    if program_graph is not None:
        pg = program_graph
        pos = pg.position
        ilp = pg.inv_log_pos
        psq = pg.position_sq
        graph = _build_graph(pg.all_dims, pg.all_lookups, ilp)
    else:
        pos = _default_position
        ilp = _default_inv_log_pos
        psq = _default_position_sq
        graph = _build_graph()
    ops = graph["ops"]
    od = graph["op_deps"]
    tt = graph.get("tight_to", {})
    produced = graph["produced"]
    consumers = graph["consumers"]
    dim_to_op = graph["dim_to_op"]

    all_dims = _all_result_dims(graph)

    output_dims = set()
    for expr in output_tokens.values():
        if isinstance(expr, Expression):
            output_dims |= set(expr.terms.keys())

    N = max_layers or _min_layers(ops, od)
    P = 4 * N
    _log(f"MILP: {len(ops)} ops, {len(all_dims)} dims, {N} layers, {P} phases")

    # ── MILP ──────────────────────────────────────────────────────
    prob = LpProblem("schedule", LpMinimize)
    D_half = LpVariable("D_half", 0, cat="Integer")
    prob += D_half

    k = {op: LpVariable(f"k_{i}", 0, N - 1, LpInteger) for i, op in enumerate(ops)}
    z = {
        op: LpVariable(f"z_{i}", 0, 1, LpBinary)
        for i, op in enumerate(ops)
        if isinstance(op, PersistDimension)
    }

    def phase_of(op):
        if isinstance(op, LookUp):
            return 4 * k[op]
        if isinstance(op, ReGLUDimension):
            return 4 * k[op] + 2
        return 4 * k[op] + 1 + 2 * z[op]

    for op in ops:
        for dep in od.get(op, set()):
            if dep in k:
                prob += phase_of(op) >= phase_of(dep) + 1

    for op, lus in tt.items():
        for lu in lus:
            if lu in k and op in k:
                prob += k[op] == k[lu]

    # death[d] = max phase of any consumer (LP variable)
    death = {}
    for d in all_dims:
        if d in output_dims:
            continue
        cons = [c_op for c_op in consumers.get(d, set()) if c_op in k]
        if not cons and d is not pos:
            continue
        dv = LpVariable(f"d_{id(d)}", 0, P - 1, LpInteger)
        for c_op in cons:
            prob += dv >= phase_of(c_op)
        death[d] = dv

    # Position must survive until any persist1 phase (for passthrough keys).
    # persist2 uses FFN passthrough (no hull key), so no position needed.
    if pos in death:
        for op in ops:
            if isinstance(op, PersistDimension) and op in z:
                prob += death[pos] >= phase_of(op) - P * z[op]

    # ── FFN width limit: at most max_ffn ReGLUs per layer ─────────
    if max_ffn is not None:
        reglus_list = graph["reglus"]
        n_reg = len(reglus_list)
        fb = {}
        for i, rg in enumerate(reglus_list):
            for L in range(N):
                fb[i, L] = LpVariable(f"fb_{i}_{L}", 0, 1, LpBinary)
            prob += lpSum(fb[i, L] for L in range(N)) == 1
            prob += k[rg] == lpSum(L * fb[i, L] for L in range(N))
        for L in range(N):
            prob += lpSum(fb[i, L] for i in range(n_reg)) <= max_ffn

    di = {d: i for i, d in enumerate(all_dims)}
    lookups = graph["lookups"]
    persists = graph["persists"]
    deps_cache = graph["deps_cache"]

    # ── Layer indicators for lookups ─────────────────────────────
    lu_at = {}
    for lu in lookups:
        for L in range(N):
            lu_at[lu, L] = LpVariable(f"la_{id(lu)}_{L}", 0, 1, LpBinary)
        prob += lpSum(lu_at[lu, L] for L in range(N)) == 1
        prob += k[lu] == lpSum(L * lu_at[lu, L] for L in range(N))

    # ── Layer indicators for persists (persist1 vs persist2) ─────
    p_layer = {}
    p1_at = {}
    for p in persists:
        for L in range(N):
            p_layer[p, L] = LpVariable(f"pl_{id(p)}_{L}", 0, 1, LpBinary)
        prob += lpSum(p_layer[p, L] for L in range(N)) == 1
        prob += k[p] == lpSum(L * p_layer[p, L] for L in range(N))
        for L in range(N):
            p1v = LpVariable(f"p1_{id(p)}_{L}", 0, 1, LpBinary)
            prob += p1v <= p_layer[p, L]
            prob += p1v <= 1 - z[p]
            prob += p1v >= p_layer[p, L] + (1 - z[p]) - 1
            p1_at[p, L] = p1v

    # ── Passthrough indicators for attention ─────────────────────
    # For each persist1 dep d, pd[d,L]=1 iff d needs passthrough at layer L
    P1_deps = defaultdict(set)
    for p in persists:
        for d in deps_cache[p]:
            if isinstance(d, (InputDimension, LookUpDimension, ReGLUDimension, PersistDimension)):
                P1_deps[d].add(p)

    pd_var = {}
    for d, p_set in P1_deps.items():
        lu_of_d = dim_to_op.get(d) if isinstance(d, LookUpDimension) else None
        for L in range(N):
            v = LpVariable(f"pd_{id(d)}_{L}", 0, 1, LpBinary)
            prob += v <= lpSum(p1_at[p, L] for p in p_set)
            if lu_of_d is not None and lu_of_d in lu_at:
                prob += v <= 1 - lu_at[lu_of_d, L]
                for p in p_set:
                    prob += v >= p1_at[p, L] - lu_at[lu_of_d, L]
            else:
                for p in p_set:
                    prob += v >= p1_at[p, L]
            pd_var[d, L] = v

    # ── Lookup heads and dims ───────────────────────────────────
    lu_h = {lu: (len(lu.value_exprs) + 1) // 2 for lu in lookups}
    lu_d = {lu: len(lu.dims) for lu in lookups}

    # ── needs_slot indicator for lookup/reglu dims ─────────────
    # A lookup dim at phase 4L needs a slot only if death >= 4L+2
    # (i.e., it survives past the persist1 boundary at 4L+1).
    # Similarly for reglu dims at phase 4L+2: needs slot if death >= 4L+4.
    # Persist dims always need slots.
    ns = {}
    for d in all_dims:
        if d in output_dims:
            continue
        if d not in death:
            continue
        prod = dim_to_op.get(d)
        if prod is None or prod not in k:
            continue
        if isinstance(d, (LookUpDimension, ReGLUDimension)):
            ns_v = LpVariable(f"ns_{di[d]}", 0, 1, LpBinary)
            prob += death[d] >= phase_of(prod) + 2 - P * (1 - ns_v)
            prob += death[d] <= phase_of(prod) + 1 + P * ns_v
            ns[d] = ns_v

    # ── Occupied slot count at each boundary (death >= c-1) ──────
    # For dims that need a slot: occupied(c) = |{d : needs_slot AND
    #   birth <= c AND death >= c-1}|.
    # Internal dims (lookup/reglu consumed within same half-layer)
    # don't occupy slots and are excluded via needs_slot.
    #
    # Also compute alive(c) = |{d : birth <= c AND death > c}|
    # for the head count dying computation.
    #
    # Positional dims always occupy their fixed slots because
    # the embedding writes to them.
    protected_dims = {pos, ilp, psq}
    alive_sum = {}
    n_inputs = sum(1 for d in all_dims if isinstance(d, InputDimension))

    for c in range(P):
        if c % 2 == 0:
            continue
        ew = []
        alive = []
        for d in all_dims:
            prod = dim_to_op.get(d)
            is_input = isinstance(d, InputDimension)

            if d in output_dims or d in protected_dims:
                if is_input:
                    ew.append(1)
                    alive.append(1)
                elif prod in k:
                    bb = LpVariable(f"b_{di[d]}_{c}", 0, 1, LpBinary)
                    prob += phase_of(prod) <= c + P * (1 - bb)
                    prob += phase_of(prod) >= (c + 1) - P * bb
                    ew.append(bb)
                    alive.append(bb)
                continue

            if d not in death:
                continue

            if is_input:
                eu = LpVariable(f"ew_{di[d]}_{c}", 0, 1, LpBinary)
                prob += death[d] >= (c - 1) - P * (1 - eu)
                prob += death[d] <= (c - 2) + P * eu
                ew.append(eu)
                au = LpVariable(f"a_{di[d]}_{c}", 0, 1, LpBinary)
                prob += death[d] >= (c + 1) - P * (1 - au)
                prob += death[d] <= c + P * au
                alive.append(au)
            elif prod in k:
                bb = LpVariable(f"b_{di[d]}_{c}", 0, 1, LpBinary)
                prob += phase_of(prod) <= c + P * (1 - bb)
                prob += phase_of(prod) >= (c + 1) - P * bb
                eu = LpVariable(f"eu_{di[d]}_{c}", 0, 1, LpBinary)
                prob += death[d] >= (c - 1) - P * (1 - eu)
                prob += death[d] <= (c - 2) + P * eu
                ev = LpVariable(f"ew_{di[d]}_{c}", 0, 1, LpBinary)
                if d in ns:
                    prob += ev <= bb
                    prob += ev <= eu
                    prob += ev <= ns[d]
                    prob += ev >= bb + eu + ns[d] - 2
                else:
                    prob += ev <= bb
                    prob += ev <= eu
                    prob += ev >= bb + eu - 1
                ew.append(ev)
                au = LpVariable(f"au_{di[d]}_{c}", 0, 1, LpBinary)
                prob += death[d] >= (c + 1) - P * (1 - au)
                prob += death[d] <= c + P * au
                av = LpVariable(f"a_{di[d]}_{c}", 0, 1, LpBinary)
                prob += av <= bb
                prob += av <= au
                prob += av >= bb + au - 1
                alive.append(av)

        prob += 2 * D_half >= lpSum(ew)
        alive_sum[c] = lpSum(alive)

    # ── Head count constraint at each layer ──────────────────────
    # heads_L = n_lu + ceil((dying + passthrough) / 2)
    # dying_L = alive_prev - alive_cur + born_L
    for L in range(N):
        c_attn = 4 * L + 1
        c_prev = 4 * L - 1

        n_lu_L = lpSum(lu_h[lu] * lu_at[lu, L] for lu in lookups)
        pt_L = lpSum(pd_var[d, L] for d in P1_deps if (d, L) in pd_var)

        born_L = lpSum(lu_d[lu] * lu_at[lu, L] for lu in lookups) + lpSum(
            p1_at[p, L] for p in persists
        )
        prev_alive = alive_sum.get(c_prev, n_inputs)
        cur_alive = alive_sum[c_attn]
        dying_L = prev_alive - cur_alive + born_L

        prob += 2 * D_half >= 2 * n_lu_L + dying_L + pt_L

    # ── Solve ─────────────────────────────────────────────────────
    _log("Solving MILP...")
    try:
        from pulp import HiGHS

        solver = HiGHS(timeLimit=3600)
    except Exception:
        from pulp import PULP_CBC_CMD

        solver = PULP_CBC_CMD(msg=0, timeLimit=3600)
    prob.solve(solver)

    if prob.status != 1:
        raise RuntimeError(f"MILP infeasible (status={prob.status}); try more layers")

    opt_D_half = int(round(value(D_half)))
    opt_D = 2 * opt_D_half
    _log(f"MILP optimal d_model: {opt_D}")

    # ── Extract assignment & build layers ──────────────────────────
    pa = {}
    for op in ops:
        layer = int(round(value(k[op])))
        if isinstance(op, LookUp):
            pa[op] = 4 * layer
        elif isinstance(op, ReGLUDimension):
            pa[op] = 4 * layer + 2
        else:
            is_p2 = int(round(value(z[op])))
            pa[op] = 4 * layer + 1 + 2 * is_p2

    max_phase = max(pa.values())
    num_layers = max_phase // 4 + 1

    by_phase = defaultdict(list)
    for op, p in pa.items():
        by_phase[p].append(op)

    std_layers = []
    for L in range(num_layers):
        std_layers.append(
            (
                [op for op in by_phase.get(4 * L, []) if isinstance(op, LookUp)],
                [op for op in by_phase.get(4 * L + 1, []) if isinstance(op, PersistDimension)],
                [op for op in by_phase.get(4 * L + 2, []) if isinstance(op, ReGLUDimension)],
                [op for op in by_phase.get(4 * L + 3, []) if isinstance(op, PersistDimension)],
            )
        )

    # ── Birth / death / alive ─────────────────────────────────────
    dim_birth = {}
    for d in all_dims:
        if isinstance(d, InputDimension):
            dim_birth[d] = -1
        else:
            prod = dim_to_op.get(d)
            if prod and prod in pa:
                dim_birth[d] = pa[prod]

    dim_death = {}
    last_boundary = 4 * num_layers - 1
    protected_post = {pos, ilp, psq}
    for d in all_dims:
        if d in output_dims or d in protected_post:
            dim_death[d] = last_boundary + 1
            continue
        last = -1
        for c_op in consumers.get(d, set()):
            if c_op in pa:
                last = max(last, pa[c_op])
        if last >= 0:
            dim_death[d] = last
        elif d in dim_birth:
            dim_death[d] = dim_birth[d]

    def _alive_at(c):
        return frozenset(
            d
            for d in all_dims
            if d in dim_birth and d in dim_death and dim_birth[d] <= c and dim_death[d] > c
        )

    alive_after = {}
    for L in range(num_layers):
        for sp in (1, 3):
            c = 4 * L + sp
            alive_after[c] = _alive_at(c)

    # ── Linear width at persist boundaries ────────────────────────
    dim_col = {d: i for i, d in enumerate(all_dims)}
    n_dims = len(all_dims)

    def _expr_vec(expr):
        v = np.zeros(n_dims)
        if isinstance(expr, Expression):
            for dim, coeff in expr.terms.items():
                if dim in dim_col:
                    v[dim_col[dim]] = coeff
        return v

    op_vecs = {}
    for op in ops:
        if isinstance(op, PersistDimension):
            op_vecs[op] = [_expr_vec(op.expr)]
        elif isinstance(op, ReGLUDimension):
            op_vecs[op] = [_expr_vec(op.a_expr), _expr_vec(op.b_expr)]
        elif isinstance(op, LookUp):
            op_vecs[op] = [
                _expr_vec(e) for e in op.query_exprs_2d + op.key_exprs_2d + op.value_exprs
            ]
    out_vecs = [_expr_vec(e) for e in output_tokens.values()]

    lin_widths = {}
    for c in sorted(alive_after):
        past = np.array([dim_birth.get(d, max_phase + 1) <= c for d in all_dims], dtype=bool)
        rows = []
        for op in ops:
            if pa[op] > c:
                for v in op_vecs.get(op, []):
                    m = v * past
                    if np.any(m != 0):
                        rows.append(m)
        for v in out_vecs:
            m = v * past
            if np.any(m != 0):
                rows.append(m)
        if rows:
            M = np.vstack(rows)
            norms = np.linalg.norm(M, axis=1, keepdims=True)
            norms[norms == 0] = 1
            lin_widths[c] = int(np.linalg.matrix_rank(M / norms))
        else:
            lin_widths[c] = 0

    # ── Expanded lin width (persists → base expressions) ──────────
    _expand_cache = {}

    def _expand_dim(dim):
        if dim in _expand_cache:
            return _expand_cache[dim]
        if isinstance(dim, PersistDimension):
            v = np.zeros(n_dims)
            for d, c in dim.expr.terms.items():
                v += c * _expand_dim(d)
        else:
            v = np.zeros(n_dims)
            if dim in dim_col:
                v[dim_col[dim]] = 1.0
        _expand_cache[dim] = v
        return v

    def _expr_vec_expanded(expr):
        v = np.zeros(n_dims)
        if isinstance(expr, Expression):
            for dim, coeff in expr.terms.items():
                v += coeff * _expand_dim(dim)
        return v

    op_vecs_exp = {}
    for op in ops:
        if isinstance(op, PersistDimension):
            op_vecs_exp[op] = [_expr_vec_expanded(op.expr)]
        elif isinstance(op, ReGLUDimension):
            op_vecs_exp[op] = [_expr_vec_expanded(op.a_expr), _expr_vec_expanded(op.b_expr)]
        elif isinstance(op, LookUp):
            op_vecs_exp[op] = [
                _expr_vec_expanded(e) for e in op.query_exprs_2d + op.key_exprs_2d + op.value_exprs
            ]
    out_vecs_exp = [_expr_vec_expanded(e) for e in output_tokens.values()]

    exp_lin_widths = {}
    for c in sorted(alive_after):
        past = np.array([dim_birth.get(d, max_phase + 1) <= c for d in all_dims], dtype=bool)
        rows = []
        for op in ops:
            if pa[op] > c:
                for v in op_vecs_exp.get(op, []):
                    m = v * past
                    if np.any(m != 0):
                        rows.append(m)
        for v in out_vecs_exp:
            m = v * past
            if np.any(m != 0):
                rows.append(m)
        if rows:
            M = np.vstack(rows)
            norms = np.linalg.norm(M, axis=1, keepdims=True)
            norms[norms == 0] = 1
            exp_lin_widths[c] = int(np.linalg.matrix_rank(M / norms))
        else:
            exp_lin_widths[c] = 0

    # ── Report ────────────────────────────────────────────────────
    max_dep = max((len(s) for s in alive_after.values()), default=0)
    max_lin = max(lin_widths.values(), default=0)
    max_exp_lin = max(exp_lin_widths.values(), default=0)
    _log(
        f"\nSchedule: {num_layers} layers, d_model={opt_D}, "
        f"max_dep={max_dep}, max_lin={max_lin}, max_exp_lin={max_exp_lin}"
    )

    for L in range(num_layers):
        attn, p1, ffn, p2 = std_layers[L]
        c1, c3 = 4 * L + 1, 4 * L + 3
        dw1 = len(alive_after.get(c1, set()))
        lw1 = lin_widths.get(c1, 0)
        ew1 = exp_lin_widths.get(c1, 0)
        dw3 = len(alive_after.get(c3, set()))
        lw3 = lin_widths.get(c3, 0)
        ew3 = exp_lin_widths.get(c3, 0)
        _log(
            f"  L{L}: A[{len(attn)}] P1[{len(p1)}] F[{len(ffn)}] P2[{len(p2)}]  "
            f"after_attn={dw1}/{lw1}/{ew1} after_ffn={dw3}/{lw3}/{ew3}"
        )

    for L in range(num_layers - 1):
        c = 4 * L + 3
        names = sorted(d.name for d in alive_after.get(c, set()))
        _log(f"  Cut L{L}/L{L + 1} ({len(names)}): {names}")

    # ── Write plan.yaml ───────────────────────────────────────────
    _write_plan(
        "plan.yaml",
        std_layers,
        num_layers,
        opt_D,
        max_dep,
        max_lin,
        alive_after,
        lin_widths,
        produced,
        max_phase,
    )
    _log("Schedule written to plan.yaml")

    return dict(
        phase_assign=pa,
        std_layers=std_layers,
        num_layers=num_layers,
        dim_birth=dim_birth,
        dim_death=dim_death,
        alive_after=alive_after,
        lin_widths=lin_widths,
        width=opt_D,
    )


class _InlineList(list):
    """Tag for lists that should be rendered inline in YAML."""

    pass


def _compact_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def _write_plan(
    path,
    std_layers,
    num_layers,
    width,
    max_dep,
    max_lin,
    alive_after,
    lin_widths,
    produced,
    max_phase,
):
    """Write schedule to plan.yaml for inspection."""
    import yaml

    yaml.add_representer(_InlineList, _compact_representer)

    plan = {
        "summary": {
            "layers": num_layers,
            "milp_d_model": width,
            "max_dep_width": max_dep,
            "max_lin_width": max_lin,
        },
        "layers": [],
    }
    for L in range(num_layers):
        attn, p1, ffn, p2 = std_layers[L]
        c1, c3 = 4 * L + 1, 4 * L + 3

        attn_dims = []
        for lu in attn:
            attn_dims.extend(d.name for d in produced.get(lu, []))

        entry = {
            "layer": L,
            "attention": _InlineList(attn_dims),
            "persist1": _InlineList(pd.name for pd in p1),
            "after_persist1": {
                "dep_width": len(alive_after.get(c1, set())),
                "lin_width": lin_widths.get(c1, 0),
                "dims": _InlineList(sorted(d.name for d in alive_after.get(c1, set()))),
            },
            "ffn": _InlineList(rg.name for rg in ffn),
            "persist2": _InlineList(pd.name for pd in p2),
        }
        if c3 <= max_phase:
            entry["after_persist2"] = {
                "dep_width": len(alive_after.get(c3, set())),
                "lin_width": lin_widths.get(c3, 0),
                "dims": _InlineList(sorted(d.name for d in alive_after.get(c3, set()))),
            }
        plan["layers"].append(entry)

    with open(path, "w") as f:
        yaml.dump(plan, f, default_flow_style=False, sort_keys=False, width=200)


# ── Utilities (used by build_model.py) ────────────────────────────


def interval_coloring(all_dims, dim_birth, dim_death, fixed=None):
    """Greedy interval coloring: assign slots to dims with [birth, death) lifetimes.

    fixed: optional dict {dim: slot} of pre-assigned slots.
    Returns dict mapping dim -> slot.
    """
    fixed = fixed or {}
    remaining = [d for d in all_dims if d not in fixed]
    items = sorted(
        (dim_birth[d], dim_death[d], i, d)
        for i, d in enumerate(remaining)
        if d in dim_birth and d in dim_death and dim_death[d] > dim_birth[d]
    )
    slot_of = dict(fixed)
    free = []
    next_slot = max(fixed.values(), default=-1) + 1
    for d, s in fixed.items():
        if d in dim_death:
            heapq.heappush(free, (dim_death[d], s))

    for birth, death_phase, _idx, d in items:
        available = []
        while free and free[0][0] <= birth:
            available.append(heapq.heappop(free)[1])
        if available:
            slot = min(available)
            for s in available:
                if s != slot:
                    heapq.heappush(free, (birth, s))
        else:
            slot = next_slot
            next_slot += 1
        slot_of[d] = slot
        heapq.heappush(free, (death_phase, slot))

    return slot_of


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run MILP scheduler for the WASM interpreter.")
    parser.add_argument("--max-layers", type=int, default=None, help="Max transformer layers")
    parser.add_argument("--max-ffn", type=int, default=None, help="Max FFN neurons per layer")
    args = parser.parse_args()

    from transformer_vm.wasm.interpreter import build

    input_tokens, output_tokens = build()

    milp_schedule(input_tokens, output_tokens, max_layers=args.max_layers, max_ffn=args.max_ffn)


if __name__ == "__main__":
    main()
