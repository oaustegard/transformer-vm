#!/usr/bin/env python3
"""Generate reference token traces by executing WASM programs directly.

Usage:
    python -m transformer_vm.wasm.reference                  # all from manifest
    python -m transformer_vm.wasm.reference data/hello.txt   # single program
    python -m transformer_vm.wasm.reference --regen           # force regenerate
"""

from __future__ import annotations

import argparse
import logging
import os

logger = logging.getLogger(__name__)

MASK32 = 0xFFFFFFFF


def to_signed(v):
    return v - (1 << 32) if v >= (1 << 31) else v


def _add_carries(a, b):
    carries = []
    carry = 0
    for i in range(4):
        s = ((a >> (8 * i)) & 0xFF) + ((b >> (8 * i)) & 0xFF) + carry
        carry = 1 if s >= 256 else 0
        carries.append(carry)
    return carries


def _sub_borrows(a, b):
    borrows = []
    borrow = 0
    for i in range(4):
        s = ((a >> (8 * i)) & 0xFF) - ((b >> (8 * i)) & 0xFF) - borrow
        borrow = 1 if s < 0 else 0
        borrows.append(borrow)
    return borrows


def _byte_tokens(value, num_bytes, carries=None):
    tokens = []
    for i in range(num_bytes):
        bv = (value >> (8 * i)) & 0xFF
        c = carries[i] if carries else 0
        tokens.append(f"{bv:02x}'" if c else f"{bv:02x}")
    return tokens


def _out_token(bv):
    bv &= 0xFF
    if 0x20 < bv < 0x7F:
        return f"out({chr(bv)})"
    return f"out({bv:02x})"


def _commit(sd, sts, bt):
    return f"commit({sd:+d},sts={sts},bt={bt})"


# ── Program loading ──────────────────────────────────────────────


def load_program(path):
    """Parse a .txt program file into a list of (op, imm) tuples and input string."""
    with open(path) as f:
        tokens = f.read().split()
    program = _parse_program_tokens(tokens)
    input_str = _extract_input(tokens)
    return program, input_str


def load_program_from_string(prefix_str):
    """Parse a prefix string into a list of (op, imm) tuples."""
    return _parse_program_tokens(prefix_str.split())


def _parse_program_tokens(tokens):
    """Parse token list with { ... } delimiters into (op, imm) pairs."""
    assert tokens[0] == "{"
    try:
        end = len(tokens) - 1 - tokens[::-1].index("}")
    except ValueError:
        raise ValueError("No closing '}' found in program") from None
    body = tokens[1:end]
    program = []
    i = 0
    while i < len(body):
        op = body[i]
        b = [int(body[i + 1 + j], 16) for j in range(4)]
        imm = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)
        program.append((op, imm))
        i += 5
    return program


def _extract_input(tokens):
    """Extract the input string from tokens after the closing '}'.

    The input section is: <byte_tokens...> commit(+0,sts=0,bt=0)
    where byte tokens are either single ASCII chars or 2-digit hex values.
    Returns the decoded string (without the trailing NUL).
    """
    try:
        end = len(tokens) - 1 - tokens[::-1].index("}")
    except ValueError:
        return ""
    input_tokens = tokens[end + 1 :]
    if not input_tokens:
        return ""
    # Drop the trailing commit(...) token
    if input_tokens and input_tokens[-1].startswith("commit("):
        input_tokens = input_tokens[:-1]
    # Decode byte tokens back to string, stopping at NUL
    chars = []
    for tok in input_tokens:
        if len(tok) == 1:
            chars.append(tok)
        elif len(tok) == 2:
            b = int(tok, 16)
            if b == 0:
                break
            chars.append(chr(b))
        else:
            break
    return "".join(chars)


# ── WASM interpreter ─────────────────────────────────────────────


def run(program, input_str="", max_tokens=1_000_000, input_base=None, trace=False):
    """Execute a compiled WASM program.

    Returns (instr_count, token_count, output_str) or, if trace=True,
    (instr_count, token_count, output_str, trace_tokens).
    """
    mem = bytearray(10 * 1024 * 1024)

    if input_base is None and program and program[0][0] == "input_base":
        input_base = program[0][1]

    if input_base is not None and input_str:
        for i, ch in enumerate(input_str.encode("utf-8") + b"\x00"):
            mem[input_base + i] = ch

    stack = []
    locals_ = [0] * 256
    call_stack = []
    pc = 0
    instr_count = 0
    token_count = 0
    output = []
    trace_tokens = [] if trace else None

    while pc < len(program) and token_count < max_tokens:
        op, imm = program[pc]
        instr_count += 1

        if op == "input_base":
            input_bytes = input_str.encode("utf-8") + b"\x00" if input_str else b"\x00"
            token_count += len(input_bytes) + 1
            if trace:
                for ch_byte in input_bytes:
                    if 0x20 < ch_byte < 0x7F:
                        trace_tokens.append(chr(ch_byte))
                    else:
                        trace_tokens.append(f"{ch_byte:02x}")
                trace_tokens.append(_commit(0, 0, 0))
            pc += 1

        elif op == "halt":
            token_count += 1
            if trace:
                trace_tokens.append("halt")
            break

        elif op == "i32.const":
            result = imm & MASK32
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(+1, 1, 0))
            pc += 1

        elif op == "local.get":
            result = locals_[imm] & MASK32
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(+1, 1, 0))
            pc += 1

        elif op == "local.set":
            val = stack.pop() & MASK32
            locals_[imm] = val
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(val, 4))
                trace_tokens.append(_commit(-1, 0, 0))
            pc += 1

        elif op == "local.tee":
            val = stack[-1] & MASK32
            locals_[imm] = val
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(val, 4))
                trace_tokens.append(_commit(0, 0, 0))
            pc += 1

        elif op == "global.get":
            result = locals_[imm] & MASK32
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(+1, 1, 0))
            pc += 1

        elif op == "global.set":
            locals_[imm] = stack.pop() & MASK32
            token_count += 1
            if trace:
                trace_tokens.append(_commit(-1, 0, 0))
            pc += 1

        elif op == "drop":
            stack.pop()
            token_count += 1
            if trace:
                trace_tokens.append(_commit(-1, 0, 0))
            pc += 1

        elif op == "select":
            c = stack.pop()
            b = stack.pop()
            a = stack.pop()
            result = a if c != 0 else b
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-2, 1, 0))
            pc += 1

        elif op == "i32.add":
            bv = stack.pop()
            av = stack.pop()
            result = (av + bv) & MASK32
            stack.append(result)
            carries = _add_carries(av, bv)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4, carries))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif op == "i32.sub":
            bv = stack.pop()
            av = stack.pop()
            result = (av - bv) & MASK32
            stack.append(result)
            borrows = _sub_borrows(av, bv)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4, borrows))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif op == "i32.eqz":
            v = stack.pop()
            result = 1 if v == 0 else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(0, 1, 0))
            pc += 1

        elif op == "i32.eq":
            bv = stack.pop()
            av = stack.pop()
            result = 1 if av == bv else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif op == "i32.ne":
            bv = stack.pop()
            av = stack.pop()
            result = 1 if av != bv else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif op == "i32.lt_s":
            bv = to_signed(stack.pop())
            av = to_signed(stack.pop())
            result = 1 if av < bv else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif op == "i32.lt_u":
            bv = stack.pop()
            av = stack.pop()
            result = 1 if av < bv else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif op == "i32.gt_s":
            bv = to_signed(stack.pop())
            av = to_signed(stack.pop())
            result = 1 if av > bv else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif op == "i32.gt_u":
            bv = stack.pop()
            av = stack.pop()
            result = 1 if av > bv else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif op == "i32.le_s":
            bv = to_signed(stack.pop())
            av = to_signed(stack.pop())
            result = 1 if av <= bv else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif op == "i32.le_u":
            bv = stack.pop()
            av = stack.pop()
            result = 1 if av <= bv else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif op == "i32.ge_s":
            bv = to_signed(stack.pop())
            av = to_signed(stack.pop())
            result = 1 if av >= bv else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif op == "i32.ge_u":
            bv = stack.pop()
            av = stack.pop()
            result = 1 if av >= bv else 0
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(-1, 1, 0))
            pc += 1

        elif op == "i32.load":
            addr = (stack.pop() + imm) & MASK32
            val = mem[addr] | (mem[addr + 1] << 8) | (mem[addr + 2] << 16) | (mem[addr + 3] << 24)
            result = val & MASK32
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(0, 1, 0))
            pc += 1

        elif op == "i32.load8_u":
            addr = (stack.pop() + imm) & MASK32
            result = mem[addr]
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(0, 1, 0))
            pc += 1

        elif op == "i32.load8_s":
            addr = (stack.pop() + imm) & MASK32
            v = mem[addr]
            result = (v - 256) & MASK32 if v >= 128 else v
            stack.append(result)
            sign = 1 if v >= 128 else 0
            carries = [sign, sign, sign, sign]
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4, carries))
                trace_tokens.append(_commit(0, 1, 0))
            pc += 1

        elif op == "i32.load16_u":
            addr = (stack.pop() + imm) & MASK32
            result = mem[addr] | (mem[addr + 1] << 8)
            stack.append(result)
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(0, 1, 0))
            pc += 1

        elif op == "i32.load16_s":
            addr = (stack.pop() + imm) & MASK32
            v = mem[addr] | (mem[addr + 1] << 8)
            result = (v - 65536) & MASK32 if v >= 32768 else v
            stack.append(result)
            sign = 1 if v >= 32768 else 0
            carries = [0, sign, sign, sign]
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4, carries))
                trace_tokens.append(_commit(0, 1, 0))
            pc += 1

        elif op == "i32.store":
            val = stack.pop()
            addr = (stack.pop() + imm) & MASK32
            mem[addr] = val & 0xFF
            mem[addr + 1] = (val >> 8) & 0xFF
            mem[addr + 2] = (val >> 16) & 0xFF
            mem[addr + 3] = (val >> 24) & 0xFF
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(val, 4))
                trace_tokens.append(_commit(-2, 0, 0))
            pc += 1

        elif op == "i32.store8":
            val = stack.pop()
            addr = (stack.pop() + imm) & MASK32
            mem[addr] = val & 0xFF
            token_count += 2
            if trace:
                trace_tokens.extend(_byte_tokens(val, 1))
                trace_tokens.append(_commit(-2, 0, 0))
            pc += 1

        elif op == "i32.store16":
            val = stack.pop()
            addr = (stack.pop() + imm) & MASK32
            mem[addr] = val & 0xFF
            mem[addr + 1] = (val >> 8) & 0xFF
            token_count += 3
            if trace:
                trace_tokens.extend(_byte_tokens(val, 2))
                trace_tokens.append(_commit(-2, 0, 0))
            pc += 1

        elif op == "br":
            offset = to_signed(imm)
            token_count += 6
            if trace:
                trace_tokens.append("branch_taken")
                trace_tokens.extend(_byte_tokens(imm & MASK32, 4))
                trace_tokens.append(_commit(0, 0, 1))
            pc = pc + 1 + offset

        elif op == "br_if":
            cond = stack.pop()
            if cond != 0:
                offset = to_signed(imm)
                pc = pc + 1 + offset
                token_count += 6
                if trace:
                    trace_tokens.append("branch_taken")
                    trace_tokens.extend(_byte_tokens(imm & MASK32, 4))
                    trace_tokens.append(_commit(-1, 0, 1))
            else:
                pc += 1
                token_count += 1
                if trace:
                    trace_tokens.append(_commit(-1, 0, 0))

        elif op == "call":
            offset = to_signed(imm)
            call_stack.append((pc + 1, list(locals_), imm & MASK32))
            locals_ = [0] * 256
            token_count += 6
            if trace:
                trace_tokens.append("branch_taken")
                trace_tokens.extend(_byte_tokens(imm & MASK32, 4))
                trace_tokens.append("call_commit")
            pc = pc + 1 + offset

        elif op == "return":
            ret_pc, ret_locals, call_imm = call_stack.pop()
            ret_offset = (ret_pc - pc - 1) & MASK32
            borrows = _sub_borrows(imm & MASK32, call_imm)
            token_count += 6
            if trace:
                trace_tokens.append("branch_taken")
                trace_tokens.extend(_byte_tokens(ret_offset, 4, borrows))
                trace_tokens.append("return_commit")
            pc = ret_pc
            locals_ = ret_locals

        elif op == "output":
            val = stack.pop() & 0xFF
            output.append(chr(val))
            token_count += 1
            if trace:
                trace_tokens.append(_out_token(val))
            pc += 1

        else:
            raise RuntimeError(f"Unknown op: {op} at pc={pc}")

    result = (instr_count, token_count, "".join(output))
    if trace:
        return result + (trace_tokens,)
    return result


# ── Trace formatting ─────────────────────────────────────────────


def format_trace(program_path, trace_tokens):
    """Format program prefix + trace tokens into reference file format."""
    with open(program_path) as f:
        raw = f.read().split()
    assert raw[0] == "{"
    end = len(raw) - 1 - raw[::-1].index("}")

    lines = ["{"]
    for i in range(1, end):
        group_idx = (i - 1) % 5
        if group_idx == 0:
            current = [raw[i]]
        else:
            current.append(raw[i])
        if group_idx == 4:
            lines.append(" ".join(current))
    lines.append("}")

    current = []
    for tok in trace_tokens:
        is_terminal = (
            tok.startswith("commit(")
            or tok.startswith("out(")
            or tok == "halt"
            or tok == "branch_taken"
            or tok == "call_commit"
            or tok == "return_commit"
        )
        current.append(tok)
        if is_terminal:
            lines.append(" ".join(current))
            current = []

    if current:
        lines.append(" ".join(current))
    return "\n".join(lines) + "\n"


# ── Reference file generation ────────────────────────────────────


def generate_ref(prog_path, ref_path=None, max_tokens=100_000_000):
    """Run a program and write the reference trace to ref_path.

    The input string is extracted from the .txt file itself (tokens after '}').

    Args:
        prog_path: Path to the .txt program file.
        ref_path: Path to write the _ref.txt output (default: <name>_ref.txt).
        max_tokens: Safety limit on trace length.
    """
    if ref_path is None:
        ref_path = prog_path.replace(".txt", "_ref.txt")
    program, input_str = load_program(prog_path)
    _instrs, token_count, output, trace_tokens = run(
        program, input_str, max_tokens=max_tokens, trace=True
    )
    formatted = format_trace(prog_path, trace_tokens)
    with open(ref_path, "w") as f:
        f.write(formatted)
    logger.info("%s: %d tokens, output=%r", ref_path, token_count, output)


def generate_all(regen=False):
    """Generate _ref.txt for all program .txt files in data/.

    Skips programs that already have a _ref.txt unless regen=True.
    """
    import glob as globmod

    from transformer_vm._paths import DATA_DIR

    prog_files = sorted(globmod.glob(os.path.join(DATA_DIR, "*.txt")))
    prog_files = [f for f in prog_files if not (f.endswith("_ref.txt") or f.endswith("_spec.txt"))]

    for prog_path in prog_files:
        ref_path = prog_path.replace(".txt", "_ref.txt")

        if not regen and os.path.exists(ref_path):
            logger.info("Skipping %s: already exists", ref_path)
            continue

        logger.info("Generating %s ...", ref_path)
        generate_ref(prog_path, ref_path)

    logger.info("Done.")


# ── CLI ──────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Generate reference token traces from WASM execution."
    )
    parser.add_argument("files", nargs="*", help="Program .txt files (default: all from data/)")
    parser.add_argument(
        "--regen", action="store_true", help="Regenerate even if _ref.txt already exists"
    )
    args = parser.parse_args()

    if not args.files:
        generate_all(regen=args.regen)
        return

    for prog_path in args.files:
        generate_ref(prog_path)


if __name__ == "__main__":
    main()
