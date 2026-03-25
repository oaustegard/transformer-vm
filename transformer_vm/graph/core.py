"""Computation graph primitives for transformer compilation.

Any program expressible as an Append-Only Lookup Machine (ALM) can be
built using these primitives and compiled into a transformer via
milp_schedule.py + build_model.py.
"""

BIG = 1e30  # large constant used to effectively zero-out attention keys (clear_key mechanism)
KEY_OFFSET = 0  # offset applied to all attention keys (for numerical stability tuning)

_all_dims = []
_all_lookups = []
_multiply_cache = {}
_reglu_cache = {}
_stepglu_cache = {}
_clear_key_cache = {}


def _expr_key(expr):
    return tuple(sorted((id(d), c) for d, c in expr.terms.items()))


class Expression:
    __slots__ = ("terms",)

    def __init__(self, terms=None):
        if terms is None:
            self.terms = {}
        elif isinstance(terms, dict):
            self.terms = {k: v for k, v in terms.items() if v != 0}
        else:
            raise TypeError

    def copy(self):
        return Expression(dict(self.terms))

    def __add__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                return self.copy()
            return self + Expression({one: other})
        if isinstance(other, Dimension):
            other = Expression({other: 1})
        if isinstance(other, Expression):
            r = dict(self.terms)
            for d, c in other.terms.items():
                r[d] = r.get(d, 0) + c
                if r[d] == 0:
                    del r[d]
            return Expression(r)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                return self.copy()
            return Expression({one: other}) + self
        if isinstance(other, Dimension):
            return Expression({other: 1}) + self
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self + (-other)
        if isinstance(other, Dimension):
            return self + Expression({other: -1})
        if isinstance(other, Expression):
            r = dict(self.terms)
            for d, c in other.terms.items():
                r[d] = r.get(d, 0) - c
                if r[d] == 0:
                    del r[d]
            return Expression(r)
        return NotImplemented

    def __rsub__(self, other):
        neg = Expression({d: -c for d, c in self.terms.items()})
        if isinstance(other, (int, float)):
            return neg + other
        if isinstance(other, Dimension):
            return Expression({other: 1}) + neg
        if isinstance(other, Expression):
            return other + neg
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                return Expression()
            return Expression({d: c * other for d, c in self.terms.items()})
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        return NotImplemented

    def __neg__(self):
        return Expression({d: -c for d, c in self.terms.items()})

    def __getitem__(self, dim):
        return self.terms.get(dim, 0)

    def __setitem__(self, dim, value):
        if value == 0 and dim in self.terms:
            del self.terms[dim]
        elif value != 0:
            self.terms[dim] = value

    def evaluate(self, values):
        return sum(c * values.get(d, 0.0) for d, c in self.terms.items())


class Dimension:
    _counter = 0

    def __init__(self, name=None, kind="generic"):
        self.id = Dimension._counter
        Dimension._counter += 1
        self.name = name or f"dim_{self.id}"
        self.kind = kind
        _all_dims.append(self)

    def _as_expr(self):
        return Expression({self: 1})

    def __add__(self, o):
        return self._as_expr().__add__(o)

    def __radd__(self, o):
        return self._as_expr().__radd__(o)

    def __sub__(self, o):
        return self._as_expr().__sub__(o)

    def __rsub__(self, o):
        return self._as_expr().__rsub__(o)

    def __mul__(self, o):
        if isinstance(o, (int, float)):
            if o == 0:
                return Expression()
            return Expression({self: o})
        return NotImplemented

    def __rmul__(self, o):
        if isinstance(o, (int, float)):
            if o == 0:
                return Expression()
            return Expression({self: o})
        return NotImplemented

    def __neg__(self):
        return Expression({self: -1})

    def __repr__(self):
        return f"{self.kind}:{self.name}[{self.id}]"


class InputDimension(Dimension):
    def __init__(self, name):
        super().__init__(name, kind="input")


one = InputDimension("one")
position = InputDimension("position")
inv_log_pos = InputDimension("inv_log_pos")
position_sq = InputDimension("position_sq")

LATEST_ALPHA = 0.3  # tie-break weight favoring more recent tokens in hardmax attention


class CumSumDimension(Dimension):
    def __init__(self, value_expr, name=None):
        super().__init__(name or f"cumsum_{Dimension._counter}", kind="cumsum")
        self.value_expr = value_expr


class PersistDimension(Dimension):
    """A dimension that stores a linear combination in a dedicated slot.

    Unlike ReGLU (which gates), persist is a pure linear projection realized
    through ff_out. It groups multiple dims into one slot, reducing d_model.
    Treated as a schedulable gate (own phase) for pathwidth/NL purposes.
    """

    def __init__(self, expr, name=None):
        super().__init__(name or f"persist_{Dimension._counter}", kind="persist")
        self.expr = expr


class ReGLUDimension(Dimension):
    def __init__(self, a_expr, b_expr, name=None):
        super().__init__(name or f"reglu_{Dimension._counter}", kind="reglu")
        self.a_expr = a_expr
        self.b_expr = b_expr


class LookUp:
    _counter = 0

    def __init__(self, value_exprs, query_exprs_2d, key_exprs_2d, tie_break="latest"):
        self.id = LookUp._counter
        LookUp._counter += 1
        self.name = None
        self.value_exprs = value_exprs
        self.query_exprs_2d = query_exprs_2d  # [qx_expr, qy_expr]
        self.key_exprs_2d = key_exprs_2d  # [kx_expr, ky_expr]
        self.tie_break = tie_break
        self.dims = [LookUpDimension(self, i) for i in range(len(value_exprs))]
        _all_lookups.append(self)


class LookUpDimension(Dimension):
    def __init__(self, lookup, value_index):
        super().__init__(f"lookup_{lookup.id}_v{value_index}", kind="lookup")
        self.lookup = lookup
        self.value_index = value_index


def _make_multiply(a, b):
    ka, kb = _expr_key(a), _expr_key(b)
    key = (ka, kb)
    if key in _multiply_cache:
        return _multiply_cache[key]
    neg_b = Expression({d: -c for d, c in b.terms.items()})
    r1 = ReGLUDimension(a, b)
    r2 = ReGLUDimension(a, neg_b)
    result = persist(Expression({r1: 1, r2: -1}))
    _multiply_cache[key] = result
    return result


def _to_expr(x):
    if isinstance(x, Expression):
        return x
    if isinstance(x, Dimension):
        return Expression({x: 1})
    if isinstance(x, (int, float)):
        if x == 0:
            return Expression()
        return Expression({one: x})
    raise TypeError(f"Cannot convert {type(x)} to Expression")


def reglu(a, b):
    """relu(b) * a — single ReGLU dimension.

    Use when b is known non-negative, giving reglu(a, b) = a * b.
    """
    a_expr = _to_expr(a)
    b_expr = _to_expr(b)
    key = (_expr_key(a_expr), _expr_key(b_expr))
    if key in _reglu_cache:
        return Expression({_reglu_cache[key]: 1})
    r = ReGLUDimension(a_expr, b_expr)
    _reglu_cache[key] = r
    return Expression({r: 1})


def stepglu(a, b):
    """a * step(b >= 0) — two ReGLU dims combined via persist.

    Equals reglu(a, b + 1) - reglu(a, b).
    For integer b: equals a when b >= 0, 0 when b < 0.
    The two ReGLU dims are internal to the FFN; the persist dim
    stores the difference in a single residual slot.
    """
    a_expr = _to_expr(a)
    b_expr = _to_expr(b)
    key = (_expr_key(a_expr), _expr_key(b_expr))
    if key in _stepglu_cache:
        return _stepglu_cache[key]
    r1 = ReGLUDimension(a_expr, b_expr + Expression({one: 1}))
    r2 = ReGLUDimension(a_expr, b_expr)
    result = persist(Expression({r1: 1, r2: -1}))
    _stepglu_cache[key] = result
    return result


def persist(expr, name=None):
    """Materialize a linear expression into a dedicated slot.

    Creates a PersistDimension that stores the value of expr. This reduces
    d_model by allowing the expression's constituents to die earlier.
    Treated as a schedulable operation (separate phase) for scheduling.
    """
    expr = _to_expr(expr)
    dim = PersistDimension(expr, name=name)
    return Expression({dim: 1})


def _to_2d_key(k, clear_key_expr=None, tie_break="latest"):
    """Map 1D key + optional clear_key to 2D key expressions via ReGLU."""
    one_expr = Expression({one: 1})
    if len(k.terms) == 1 and one in k.terms:
        c = k.terms[one]
        k_abs = Expression({one: c * c})
    elif len(k.terms) == 1 and position in k.terms:
        c = k.terms[position]
        k_abs = Expression({position_sq: c * c})
    else:
        k_abs = _make_multiply(k, k)
    kx = k * 2 - one_expr * (2 * KEY_OFFSET)
    ky = -k_abs + k * (2 * KEY_OFFSET) - one_expr * (KEY_OFFSET**2)
    if clear_key_expr is not None:
        if len(clear_key_expr.terms) == 1:
            clear = clear_key_expr
        else:
            ck_key = _expr_key(clear_key_expr)
            if ck_key not in _clear_key_cache:
                _clear_key_cache[ck_key] = persist(clear_key_expr)
            clear = _clear_key_cache[ck_key]
        ky = ky - clear * BIG
    if tie_break == "latest":
        ky = ky + Expression({inv_log_pos: LATEST_ALPHA})
    elif tie_break == "average":
        ky = Expression({one: 1})
    return [kx, ky]


def _to_2d_query(q):
    """Map 1D query to 2D query expressions (purely linear)."""
    one_expr = Expression({one: 1})
    return [q - one_expr * KEY_OFFSET, one_expr]


def fetch(value, query=None, key=None, clear_key=None, tie_break="latest"):
    is_list = isinstance(value, (list, tuple))
    values = list(value) if is_list else [value]
    value_exprs = [_to_expr(v) for v in values]
    q = _to_expr(query) if query is not None else Expression()
    k = _to_expr(key) if key is not None else Expression()
    ck = _to_expr(clear_key) if clear_key is not None else None
    key_2d = _to_2d_key(k, ck, tie_break=tie_break)
    query_2d = _to_2d_query(q)
    lookup = LookUp(value_exprs, query_2d, key_2d, tie_break=tie_break)
    if is_list:
        return tuple(lookup.dims)
    return lookup.dims[0]


def _name_expr_dims(name, expr):
    """Name ReGLU, Persist and LookUp dims embedded in an Expression."""
    if not isinstance(expr, Expression):
        return
    pos_idx = neg_idx = lu_idx = persist_idx = 0
    for dim, coeff in expr.terms.items():
        if isinstance(dim, PersistDimension) and dim.name.startswith("persist_"):
            dim.name = name if persist_idx == 0 else f"{name}${persist_idx}"
            persist_idx += 1
        elif isinstance(dim, ReGLUDimension) and dim.name.startswith("reglu_"):
            if coeff > 0:
                dim.name = f"{name}+" if pos_idx == 0 else f"{name}+{pos_idx}"
                pos_idx += 1
            else:
                dim.name = f"{name}-" if neg_idx == 0 else f"{name}-{neg_idx}"
                neg_idx += 1
        elif isinstance(dim, LookUpDimension) and dim.name.startswith("lookup_"):
            dim.name = f"{name}_lu" if lu_idx == 0 else f"{name}_lu{lu_idx}"
            if dim.lookup.name is None:
                dim.lookup.name = dim.name
            lu_idx += 1


def auto_name(local_vars):
    """Name graph nodes based on variable names from caller's locals().

    Call as auto_name(locals()) at the end of build() to give meaningful
    names to LookUps, ReGLUDimensions, etc. for diagnostic output.
    """
    for name, val in local_vars.items():
        if name.startswith("_"):
            continue
        if isinstance(val, Dimension) and not isinstance(val, InputDimension):
            val.name = name
            if isinstance(val, LookUpDimension) and val.lookup.name is None:
                val.lookup.name = name
        elif isinstance(val, Expression):
            _name_expr_dims(name, val)
        elif isinstance(val, (list, tuple)):
            for i, item in enumerate(val):
                if isinstance(item, Dimension) and not isinstance(item, InputDimension):
                    item.name = f"{name}[{i}]"
                    if isinstance(item, LookUpDimension) and item.lookup.name is None:
                        item.lookup.name = name
                elif isinstance(item, Expression):
                    _name_expr_dims(f"{name}[{i}]", item)


def fetch_sum(value_list):
    """Cumulative sum via attention averaging: avg * position.

    Position 0 (start token, one=0) is excluded from the average
    because its ky=0 < 1, so the denominator is p (not p+1).
    Multiplying by position recovers the exact cumulative sum.
    """
    if not isinstance(value_list, (list, tuple)):
        value_list = [value_list]
    key = Expression({one: KEY_OFFSET})
    query = Expression({one: KEY_OFFSET})
    avg_dims = fetch(value_list, query=query, key=key, tie_break="average")
    if not isinstance(avg_dims, tuple):
        avg_dims = (avg_dims,)
    results = [reglu(_to_expr(d), _to_expr(position)) for d in avg_dims]
    return tuple(results) if len(results) > 1 else results[0]


# ── Graph lifecycle ──────────────────────────────────────────────────


def reset_graph():
    """Reset all graph state for building a new program.

    Must be called before constructing a new computation graph.
    Recreates the built-in positional dimensions (one, position,
    inv_log_pos, position_sq) that every ALM requires.
    """
    global one, position, inv_log_pos, position_sq
    _all_dims.clear()
    _all_lookups.clear()
    _multiply_cache.clear()
    _reglu_cache.clear()
    _stepglu_cache.clear()
    _clear_key_cache.clear()
    Dimension._counter = 0
    LookUp._counter = 0
    one = InputDimension("one")
    position = InputDimension("position")
    inv_log_pos = InputDimension("inv_log_pos")
    position_sq = InputDimension("position_sq")


class ProgramGraph:
    """Captured computation graph for a program, ready for scheduling
    and weight construction.

    Created by Program.build() after the graph is fully constructed.
    """

    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.all_dims = list(_all_dims)
        self.all_lookups = list(_all_lookups)
        self.one = one
        self.position = position
        self.inv_log_pos = inv_log_pos
        self.position_sq = position_sq
