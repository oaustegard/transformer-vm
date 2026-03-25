from transformer_vm.graph import core as _graph
from transformer_vm.graph.core import (
    Expression,
    InputDimension,
    ReGLUDimension,
    auto_name,
    fetch,
    fetch_sum,
    persist,
    reglu,
    stepglu,
)

# Squared radius of the circle on which opcode dispatch points lie
# (all points satisfy x² + y² = 32045)
pointsR2 = 32045
# Circle points for opcode dispatch; each opcode maps to a unique point,
# enabling single-neuron opcode detection via dot product
points = [
    (179, 2),
    (179, -2),
    (-179, 2),
    (-179, -2),
    (2, 179),
    (2, -179),
    (-2, 179),
    (-2, -179),
    (178, 19),
    (178, -19),
    (-178, 19),
    (-178, -19),
    (19, 178),
    (19, -178),
    (-19, 178),
    (-19, -178),
    (173, 46),
    (173, -46),
    (-173, 46),
    (-173, -46),
    (46, 173),
    (46, -173),
    (-46, 173),
    (-46, -173),
    (166, 67),
    (166, -67),
    (-166, 67),
    (-166, -67),
    (67, 166),
    (67, -166),
    (-67, 166),
    (-67, -166),
    (163, 74),
    (163, -74),
    (-163, 74),
    (-163, -74),
    (74, 163),
    (74, -163),
    (-74, 163),
    (-74, -163),
    (157, 86),
    (157, -86),
    (-157, 86),
    (-157, -86),
    (86, 157),
    (86, -157),
    (-86, 157),
    (-86, -157),
    (142, 109),
    (142, -109),
    (-142, 109),
    (-142, -109),
    (109, 142),
    (109, -142),
    (-109, 142),
    (-109, -142),
    (131, 122),
    (131, -122),
    (-131, 122),
    (-131, -122),
    (122, 131),
    (122, -131),
    (-122, 131),
    (-122, -131),
]

OPCODES = {
    "halt": 0x00,
    "return": 0x0F,
    "call": 0x10,
    "br": 0x0C,
    "br_if": 0x0D,
    "drop": 0x1A,
    "select": 0x1B,
    "local.get": 0x20,
    "local.set": 0x21,
    "local.tee": 0x22,
    "global.get": 0x23,
    "global.set": 0x24,
    "i32.load": 0x28,
    "i32.load8_s": 0x2C,
    "i32.load8_u": 0x2D,
    "i32.load16_s": 0x2E,
    "i32.load16_u": 0x2F,
    "i32.store": 0x36,
    "i32.store8": 0x3A,
    "i32.store16": 0x3B,
    "i32.const": 0x41,
    "i32.eqz": 0x45,
    "i32.eq": 0x46,
    "i32.ne": 0x47,
    "i32.lt_s": 0x48,
    "i32.lt_u": 0x49,
    "i32.gt_s": 0x4A,
    "i32.gt_u": 0x4B,
    "i32.le_s": 0x4C,
    "i32.le_u": 0x4D,
    "i32.ge_s": 0x4E,
    "i32.ge_u": 0x4F,
    "i32.add": 0x6A,
    "i32.sub": 0x6B,
    "output": 0xFF,
    "input_base": 0xFE,
}
OPCODE_POINT = {op: points[i] for i, op in enumerate(OPCODES)}


def get_byte_value(bv, i, signed=False):
    return (bv - 256 if signed and bv >= 128 else bv) * (1 << (8 * i))


STACK_DELTA = {
    "i32.const": +1,
    "drop": -1,
    "i32.add": -1,
    "i32.sub": -1,
    "i32.eqz": 0,
    "i32.gt_s": -1,
    "i32.gt_u": -1,
    "i32.le_s": -1,
    "i32.le_u": -1,
    "i32.ge_s": -1,
    "i32.ge_u": -1,
    "i32.lt_s": -1,
    "i32.lt_u": -1,
    "i32.eq": -1,
    "i32.ne": -1,
    "select": -2,
    "local.set": -1,
    "local.get": +1,
    "local.tee": 0,
    "global.get": +1,
    "global.set": -1,
    "i32.store8": -2,
    "i32.store16": -2,
    "i32.store": -2,
    "i32.load8_s": 0,
    "i32.load8_u": 0,
    "i32.load16_s": 0,
    "i32.load16_u": 0,
    "i32.load": 0,
    "br_if": -1,
    "br": 0,
    "output": -1,
    "halt": 0,
    "call": 0,
    "return": 0,
    "input_base": 0,
}
STS_OPS = {
    "i32.const",
    "i32.add",
    "i32.sub",
    "i32.eqz",
    "select",
    "i32.gt_s",
    "i32.gt_u",
    "i32.le_s",
    "i32.le_u",
    "i32.ge_s",
    "i32.ge_u",
    "i32.lt_s",
    "i32.lt_u",
    "i32.eq",
    "i32.ne",
    "local.get",
    "global.get",
    "i32.load8_s",
    "i32.load8_u",
    "i32.load16_s",
    "i32.load16_u",
    "i32.load",
}
# Address stride between local variable slots per call depth
# (each local occupies 4 bytes, 256 allows up to 64 locals)
LOCAL_STRIDE = 256


def build(program=None):
    one = _graph.one
    position = _graph.position

    # Circle-point opcode dispatch: 1 ReGLU per case
    _op_dot_cache = {}

    def op_dot(op):
        """Gate expression for opcode: 1 when op matches, <= -1 otherwise."""
        if op not in _op_dot_cache:
            px, py = OPCODE_POINT[op]
            _op_dot_cache[op] = px * fetched_opcode_x + py * fetched_opcode_y - pointsR2 * one + 1
        return _op_dot_cache[op]

    _is_op_cache = {}

    def is_op(op):
        if op not in _is_op_cache:
            if program is not None:
                _is_op_cache[op] = stepglu(one, op_dot(op))
            else:
                _is_op_cache[op] = reglu(one, op_dot(op))
        return _is_op_cache[op]

    byte_number = InputDimension("byte_number")
    carry = InputDimension("carry")
    delta_cursor = InputDimension("delta_cursor")
    delta_stack = InputDimension("delta_stack")
    is_jump = InputDimension("is_jump")
    store_to_stack = InputDimension("store_to_stack")
    is_branch_taken = InputDimension("is_branch_taken")
    delta_call_depth = InputDimension("delta_call_depth")
    is_return_commit = InputDimension("is_return_commit")

    if program is None:
        delta_stack_prefix = InputDimension("delta_stack_prefix")
        store_to_stack_prefix = InputDimension("store_to_stack_prefix")
        opcode_x = InputDimension("opcode_x")
        opcode_y = InputDimension("opcode_y")
        is_write = InputDimension("is_write")

    # ── Build input_tokens ───────────────────────────────────────────
    input_tokens = (
        {
            (f"{bv:02x}'" if c else f"{bv:02x}"): (bv + 1) * byte_number + c * carry
            for bv in range(256)
            for c in range(2)
        }
        | {
            f"commit({sd:+d},sts={sts},bt={bt})": delta_cursor
            + sd * delta_stack
            + sts * store_to_stack
            + bt * is_jump
            for sd, sts in {(STACK_DELTA[op], 1 if op in STS_OPS else 0) for op in OPCODES}
            for bt in range(2)
        }
        | {
            (f"out({chr(bv)})" if 0x20 < bv < 0x7F else f"out({bv:02x})"): delta_cursor
            * 1
            for bv in range(256)
        }
        | {
            "branch_taken": 1 * is_branch_taken,
            "call_commit": delta_cursor + delta_call_depth + is_jump,
            "return_commit": delta_cursor - delta_call_depth + is_return_commit + is_jump,
        }
    )

    if program is None:
        input_tokens["{"] = 0 * one
        input_tokens["}"] = 3 * delta_stack
        for op in OPCODES:
            sd = STACK_DELTA[op]
            sts = 1 if op in STS_OPS else 0
            px, py = OPCODE_POINT[op]
            embedding = (
                px * opcode_x
                + py * opcode_y
                + sd * delta_stack_prefix
                + sts * store_to_stack_prefix
                + (
                    1
                    if op in ("local.set", "local.tee", "i32.store8", "i32.store16", "i32.store")
                    else 0
                )
                * is_write
            )
            input_tokens[op] = embedding
    else:
        input_tokens["start"] = 3 * delta_stack

    # Printable ASCII aliases: 'a' -> same embedding as '61', etc.
    # Skip 0x20 (space) — space is not a valid token name.
    for bv in range(0x21, 0x7F):
        ch = chr(bv)
        if ch in input_tokens:
            continue
        input_tokens[ch] = input_tokens[f"{bv:02x}"]

    start_token = "start" if program is not None else "{"
    for tok in input_tokens:
        if tok != start_token:
            input_tokens[tok][one] = 1

    # ── store_value and branch offset (needed before fetch_sum) ─────

    store_bytes = [fetch(byte_number - 1, query=position - i, key=position) for i in range(1, 5)]
    store_value = sum((1 << (8 * (4 - i))) * store_bytes[i - 1] for i in range(1, 5))
    store_value = persist(store_value)
    msb = store_bytes[0]
    unsigned_branch = reglu(store_value, is_jump)
    jump_sign = stepglu(one, msb + 128 * is_jump - 256)
    delta_cursor_expr = delta_cursor + unsigned_branch - jump_sign * (1 << 32)

    byte_index = position - fetch(position, query=one, key=one, clear_key=byte_number)
    is_boundary = stepglu(one, -byte_number)

    # ── Cumulative state ─────────────────────────────────────────────
    stack_depth, cursor, call_depth = fetch_sum([delta_stack, delta_cursor_expr, delta_call_depth])

    # ── Instruction fetch ────────────────────────────────────────────
    if program is None:
        instruction_position = 5 * cursor + 1
        (
            fetched_opcode_x,
            fetched_opcode_y,
            fetched_stack_delta,
            fetched_store_to_stack,
            fetched_is_write,
        ) = fetch(
            [opcode_x, opcode_y, delta_stack_prefix, store_to_stack_prefix, is_write],
            query=instruction_position,
            key=position,
        )
        immediate = sum(
            (1 << (8 * (i - 1)))
            * fetch(byte_number - 1, query=instruction_position + i, key=position)
            for i in range(1, 5)
        )
        immediate = persist(immediate)
    else:
        N_instr = len(program)
        one_expr = Expression({one: 1})
        cursor_expr = cursor if isinstance(cursor, Expression) else Expression({cursor: 1})

        r_pos = []
        r_neg = []
        for i in range(1, N_instr):
            r_pos.append(ReGLUDimension(one_expr, cursor_expr - i + 1, name=f"step_pos_{i}"))
            r_neg.append(ReGLUDimension(one_expr, cursor_expr - i, name=f"step_neg_{i}"))

        def _cursor_lookup(values, name=None):
            """Piecewise-constant FFN lookup: cursor -> values[cursor]."""
            expr = Expression({one: values[0]})
            for i in range(1, N_instr):
                diff = values[i] - values[i - 1]
                if diff == 0:
                    continue
                expr[r_pos[i - 1]] = diff
                expr[r_neg[i - 1]] = -diff
            return persist(expr, name=name)

        WRITE_OPS = {"local.set", "local.tee", "i32.store8", "i32.store16", "i32.store"}
        opx_vals = [OPCODE_POINT[ins["opcode"]][0] for ins in program]
        opy_vals = [OPCODE_POINT[ins["opcode"]][1] for ins in program]
        sd_vals = [STACK_DELTA[ins["opcode"]] for ins in program]
        sts_vals = [1 if ins["opcode"] in STS_OPS else 0 for ins in program]
        iw_vals = [1 if ins["opcode"] in WRITE_OPS else 0 for ins in program]

        fetched_opcode_x = _cursor_lookup(opx_vals, "fetched_opcode_x")
        fetched_opcode_y = _cursor_lookup(opy_vals, "fetched_opcode_y")
        fetched_stack_delta = _cursor_lookup(sd_vals, "fetched_stack_delta")
        fetched_store_to_stack = _cursor_lookup(sts_vals, "fetched_store_to_stack")
        fetched_is_write = _cursor_lookup(iw_vals, "fetched_is_write")

        imm_vals = [sum(ins["bytes"][j] * (1 << (8 * j)) for j in range(4)) for ins in program]
        immediate = _cursor_lookup(imm_vals, "immediate")

    # Derived instruction properties
    is_output = is_op("output")
    memory_write_gate = persist(
        is_op("i32.store") + is_op("i32.store8") + is_op("i32.store16") + is_op("input_base")
    )
    uses_top_byte = fetched_is_write + is_output
    is_producing_bytes = fetched_store_to_stack + fetched_is_write

    # ── Local variables ─────────────────────────────────────────────

    local_write_key_dim = LOCAL_STRIDE * call_depth + 4 * immediate + byte_index
    not_local_write = 1 - is_op("local.set") - is_op("local.tee") + is_boundary
    local_byte = fetch(
        byte_number - 1,
        query=local_write_key_dim + 1,
        key=local_write_key_dim,
        clear_key=not_local_write,
    )

    # ── Stack access ─────────────────────────────────────────────────
    not_store_to_stack = 1 - store_to_stack
    stack_top_value, stack_top_position = fetch(
        [store_value, position - 4],
        query=stack_depth,
        key=stack_depth,
        clear_key=not_store_to_stack,
    )

    stack_second_value, stack_second_position = fetch(
        [store_value, position - 4],
        query=stack_depth - 1,
        key=stack_depth,
        clear_key=not_store_to_stack,
    )

    stack_third_position = fetch(
        position - 4, query=stack_depth - 2, key=stack_depth, clear_key=not_store_to_stack
    )

    top_byte = fetch(byte_number - 1, query=stack_top_position + byte_index, key=position)
    second_byte = fetch(byte_number - 1, query=stack_second_position + byte_index, key=position)
    third_byte = fetch(byte_number - 1, query=stack_third_position + byte_index, key=position)

    # ── Memory ───────────────────────────────────────────────────────
    memory_read_address = stack_top_value + immediate + byte_index
    memory_write_address = stack_second_value + immediate + byte_index - 1
    not_memory_write_byte = 1 + is_boundary - memory_write_gate
    memory_byte_dirty, memory_byte_dirty_position = fetch(
        [byte_number - 1, memory_write_address],
        query=memory_read_address,
        key=memory_write_address,
        clear_key=not_memory_write_byte,
    )
    diff = memory_byte_dirty_position - memory_read_address
    memory_byte = (
        reglu(memory_byte_dirty, diff + 1)
        - 2 * reglu(memory_byte_dirty, diff)
        + reglu(memory_byte_dirty, diff - 1)
    )
    memory_sign = stepglu(one, memory_byte - 128)

    # ── Arithmetic ───────────────────────────────────────────────────
    carry_late = persist(carry)

    add_value = second_byte + top_byte + carry_late
    add_carry = stepglu(one, add_value - 256)
    add_byte = add_value - 256 * add_carry

    sub_value = second_byte - top_byte - carry_late
    sub_borrow = 1 - stepglu(one, sub_value)
    sub_byte = sub_value + 256 * sub_borrow

    # ── Comparisons ──────────────────────────────────────────────────
    a_gt_b_u = stepglu(one, stack_second_value - stack_top_value - 1)
    a_lt_b_u = stepglu(one, stack_top_value - stack_second_value - 1)
    a_eq_b = one - a_gt_b_u - a_lt_b_u

    sign_diff = persist(
        reglu(one, stack_top_value - (1 << 31) + 1)
        - reglu(one, stack_top_value - (1 << 31))
        - reglu(one, stack_second_value - (1 << 31) + 1)
        + reglu(one, stack_second_value - (1 << 31))
    )
    a_gt_b_s = stepglu(one, sign_diff + a_gt_b_u - 1)
    a_lt_b_s = stepglu(one, -sign_diff + a_lt_b_u - 1)

    cond_nonzero = stepglu(one, stack_top_value - 1)

    # ── Call stack (byte-level, for CALL/RETURN) ────────────────────
    call_stack_write_key = call_depth * 4 + byte_index - 1
    call_stack_read_key = (call_depth - 1) * 4 + byte_index
    if program is not None:
        not_call_byte = 1 - stepglu(one - is_boundary, op_dot("call"))
    else:
        not_call_byte = 1 - reglu(one - is_boundary, op_dot("call"))
    call_stack_byte = fetch(
        byte_number - 1,
        query=call_stack_read_key,
        key=call_stack_write_key,
        clear_key=not_call_byte,
    )

    # ── Result byte ──────────────────────────────────────────────────
    if program is None:
        const_byte = fetch(
            byte_number - 1, query=instruction_position + byte_index + 1, key=position
        )
    else:
        cb = []
        for b in range(4):
            bvals = [ins["bytes"][b] for ins in program]
            cb.append(_cursor_lookup(bvals, f"const_byte_{b}"))
        const_byte = cb[0]
        for b in range(1, 4):
            const_byte = const_byte + stepglu(cb[b] - cb[b - 1], byte_index - b)

    # ── Branch/return byte (J token value) ────────────────────────
    csb = reglu(call_stack_byte, op_dot("return"))
    cc = reglu(carry_late, op_dot("return"))
    branch_sub_val = const_byte - csb - cc
    branch_sub_val = persist(branch_sub_val)
    branch_sub_borrow = 1 - stepglu(one, branch_sub_val)
    branch_byte = branch_sub_val + 256 * branch_sub_borrow
    branch_carry = branch_sub_borrow

    byte_at_2 = stepglu(one, byte_index - 2)

    result_byte_early = persist(
        reglu(local_byte, op_dot("local.get"))
        + reglu(top_byte, uses_top_byte)
        + reglu(third_byte, op_dot("select") + cond_nonzero - 1)
        + reglu(second_byte, op_dot("select") - cond_nonzero)
    )

    result_byte = persist(
        reglu(branch_sub_val, op_dot("i32.const"))
        + reglu(add_byte, op_dot("i32.add"))
        + reglu(sub_byte, op_dot("i32.sub"))
        + result_byte_early
        + reglu(memory_byte, op_dot("i32.load"))
        + reglu(memory_byte, op_dot("i32.load8_u") + is_boundary - 1)
        + reglu(memory_byte, op_dot("i32.load8_s") + is_boundary - 1)
        + 255 * reglu(carry_late, op_dot("i32.load8_s") - is_boundary)
        + reglu(memory_byte, op_dot("i32.load16_u") - byte_at_2)
        + reglu(memory_byte, op_dot("i32.load16_s") - byte_at_2)
        + 255 * reglu(carry_late, op_dot("i32.load16_s") + byte_at_2 - 1)
        + reglu(branch_sub_val, op_dot("br"))
        + reglu(branch_sub_val, op_dot("br_if"))
        + reglu(branch_sub_val, op_dot("call"))
        + reglu(branch_byte, op_dot("return"))
    ) + persist(
        reglu(a_eq_b, op_dot("i32.eq") + is_boundary - 1)
        + reglu(1 - a_eq_b, op_dot("i32.ne") + is_boundary - 1)
        + reglu(a_gt_b_u, op_dot("i32.gt_u") + is_boundary - 1)
        + reglu(1 - a_gt_b_u, op_dot("i32.le_u") + is_boundary - 1)
        + reglu(a_lt_b_u, op_dot("i32.lt_u") + is_boundary - 1)
        + reglu(1 - a_lt_b_u, op_dot("i32.ge_u") + is_boundary - 1)
        + reglu(a_gt_b_s, op_dot("i32.gt_s") + is_boundary - 1)
        + reglu(1 - a_gt_b_s, op_dot("i32.le_s") + is_boundary - 1)
        + reglu(a_lt_b_s, op_dot("i32.lt_s") + is_boundary - 1)
        + reglu(1 - a_lt_b_s, op_dot("i32.ge_s") + is_boundary - 1)
        + reglu(1 - cond_nonzero, op_dot("i32.eqz") + is_boundary - 1)
    )

    result_carry = persist(
        reglu(add_carry, op_dot("i32.add"))
        + reglu(sub_borrow, op_dot("i32.sub"))
        + reglu(memory_sign, op_dot("i32.load8_s") + is_boundary - 1)
        + reglu(carry_late, op_dot("i32.load8_s") - is_boundary)
        + reglu(memory_sign, op_dot("i32.load16_s") - is_boundary)
        + reglu(carry_late - memory_sign, op_dot("i32.load16_s") + byte_at_2 - 1)
        + reglu(branch_carry, op_dot("return"))
    )

    # ── Next-token prediction (linear in state) ──────────────────────
    byte_index_4 = stepglu(one, byte_index - 4)
    early_done = reglu(1 - is_boundary, op_dot("i32.store8")) + reglu(
        byte_at_2, op_dot("i32.store16")
    )
    early_done = persist(early_done)
    byte_done = byte_index_4 + early_done
    is_byte_seq = 1 - is_boundary - byte_done

    emit_halt = reglu(is_boundary, op_dot("halt"))
    emit_branch_taken = (
        reglu(is_boundary, op_dot("br") - is_branch_taken)
        + reglu(is_boundary, cond_nonzero + op_dot("br_if") - is_branch_taken - 1)
        + reglu(is_boundary, op_dot("return") - is_branch_taken)
        + reglu(is_boundary, op_dot("call") - is_branch_taken)
    )
    emit_return_commit = reglu(byte_index_4, op_dot("return"))
    emit_out = reglu(is_boundary, op_dot("output"))
    emit_byte_start = reglu(is_producing_bytes, is_boundary)
    emit_byte = emit_byte_start + is_byte_seq + is_branch_taken
    emit_call_commit = reglu(byte_index_4, op_dot("call"))
    emit_bt = reglu(byte_done, op_dot("br")) + reglu(byte_done, op_dot("br_if"))
    emit_commit = (
        byte_done
        + is_boundary
        - emit_halt
        - emit_branch_taken
        - emit_return_commit
        - emit_out
        - emit_byte_start
        - is_branch_taken
        - emit_call_commit
    )

    # ── Build output_tokens (scoring expressions for argmax prediction) ──
    H = 1e5
    output_tokens = {}
    output_tokens["halt"] = H * emit_halt
    output_tokens["branch_taken"] = H * emit_branch_taken
    output_tokens["call_commit"] = H * emit_call_commit
    output_tokens["return_commit"] = H * emit_return_commit

    for bv in range(256):
        tok = f"out({chr(bv)})" if 0x20 < bv < 0x7F else f"out({bv:02x})"
        output_tokens[tok] = H * emit_out + (2 * bv) * top_byte - bv * bv

    for sd, sts in {(STACK_DELTA[op], 1 if op in STS_OPS else 0) for op in OPCODES}:
        for bt in range(2):
            output_tokens[f"commit({sd:+d},sts={sts},bt={bt})"] = (
                H * emit_commit
                + (2 * sd) * fetched_stack_delta
                - sd * sd
                + (2 * sts) * fetched_store_to_stack
                - sts * sts
                + (2 * bt) * emit_bt
                - bt * bt
            )

    for bv in range(256):
        bv_base = H * emit_byte + (2 * bv) * result_byte - bv * bv
        for c in range(2):
            score = bv_base + (2 * c) * result_carry - c * c
            output_tokens[f"{bv:02x}'" if c else f"{bv:02x}"] = score

    auto_name(locals())
    return input_tokens, output_tokens


class WASMMachine:
    """The WASM interpreter expressed as a Program.

    When compiled via build_model(), produces the universal WASM executor
    transformer.

    With a program argument, produces a specialized transformer via the
    First Futamura Projection: the program's bytecode is baked into FFN
    weights, eliminating the program prefix and instruction-fetch attention.
    """

    def __init__(self, program=None):
        self.program = program

    def build(self):
        from transformer_vm.graph.core import ProgramGraph, reset_graph

        reset_graph()
        input_tokens, output_tokens = build(program=self.program)
        return ProgramGraph(input_tokens, output_tokens)
