# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SCF dialect op definitions.

Core ops:

  scf.for     — Bounded counted loop with bracketed range syntax.
  scf.if      — Conditional execution with both regions required.
  scf.lookup  — Keyed whole-value table selection.
  scf.select  — Scalar-condition whole-value selection.
  scf.yield   — Variadic terminator for all scf op regions.

Always-explicit yield (no implicit terminator). Every scf op region
ends with a literal scf.yield, so passes walking region terminators
have a single uniform op kind to handle.
"""

from loom.assembly import (
    ARROW,
    COLON,
    COMMA,
    EQUALS,
    LBRACKET,
    RBRACKET,
    AttrTable,
    BindingList,
    OptionalGroup,
    Ref,
    Refs,
    Region,
    ResultType,
    ResultTypeList,
    TypesOf,
    kw,
)
from loom.dsl import (
    ANY,
    ATTR_TYPE_I64_ARRAY,
    I1,
    INDEX,
    PURE,
    SAFE_TO_SPECULATE,
    TERMINATOR,
    AttrDef,
    Dialect,
    IterArgsMatchResults,
    LoopLikeInterface,
    Op,
    Operand,
    RegionBranchInterface,
    RegionDef,
    Result,
    SameType,
    YieldCountMatchesResults,
    YieldTypesMatchResults,
)

# ============================================================================
# Dialect
# ============================================================================

scf_ops = Dialect(
    "scf",
    dialect_id=0x05,
    doc="Structured control flow operations.",
)

# ============================================================================
# scf.yield — variadic region terminator
# ============================================================================
#
# Single shared terminator for every scf op region (scf.for body,
# scf.if then/else). No implicit-yield variant: regions always end
# with an explicit scf.yield, even when there are no values to
# forward. Trades one line of printed IR for uniform terminator
# handling in every pass.

scf_yield = Op(
    "scf.yield",
    group=scf_ops,
    doc="Region terminator forwarding values to the parent scf op.",
    operands=[Operand("values", ANY, variadic=True)],
    traits=[TERMINATOR],
    format=[
        OptionalGroup(
            [Refs("values"), COLON, TypesOf("values")],
            anchor="values",
        ),
    ],
    examples=[
        "scf.yield",
        "scf.yield %first, %second : f32, i32",
    ],
)

# ============================================================================
# scf.select — scalar-condition whole-value select
# ============================================================================
#
# Chooses one of two already-computed SSA values using a scalar i1 condition.
# This is deliberately different from vector.select, which is a lanewise
# vector-mask operation. scf.select is the generic result of reducing
# yield-only scf.if regions and can select scalars, address-domain values,
# vectors, tiles, views, encodings, or any other same-typed SSA value without
# making the scf dialect depend on every value dialect.

scf_select = Op(
    "scf.select",
    group=scf_ops,
    doc=(
        "Select between two same-typed SSA values using a scalar i1 condition. "
        "This is whole-value selection: if the selected values are vectors or "
        "tiles, the entire aggregate is chosen as one value. Lanewise vector "
        "masking remains vector.select."
    ),
    operands=[
        Operand("condition", I1),
        Operand("true_value", ANY),
        Operand("false_value", ANY),
    ],
    results=[Result("result", ANY)],
    constraints=[SameType("true_value", "false_value", "result")],
    traits=[PURE, SAFE_TO_SPECULATE],
    canonicalize="loom_scf_select_canonicalize",
    facts="loom_scf_select_facts",
    format=[
        Ref("condition"),
        COMMA,
        Ref("true_value"),
        COMMA,
        Ref("false_value"),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%r = scf.select %cond, %a, %b : f32",
        "%v = scf.select %cond, %then_vec, %else_vec : vector<16xf32>",
    ],
)

# ============================================================================
# scf.lookup — keyed whole-value table selection
# ============================================================================
#
# Total table lookup over already-computed SSA values. The case table names
# every explicit selector value, and every row lists one complete result tuple.
# The final `default(...)` row is selected when the selector does not equal any
# explicit key. The grouped assembly is backed by generic i64-array attrs and
# variadic operands, preserving a strict verifier contract without a custom
# op-specific parser.

scf_lookup = Op(
    "scf.lookup",
    group=scf_ops,
    doc=(
        "Total table lookup over already-computed SSA values. The selector is "
        "an index value. The table gives sorted unique explicit case rows. "
        "default(...) gives the total fallback row. The result count is the "
        "row width, and every payload value in a column must match that "
        "column's result type."
    ),
    operands=[
        Operand("selector", INDEX),
        Operand("values", ANY, variadic=True),
    ],
    results=[Result("results", ANY, variadic=True)],
    attrs=[
        AttrDef(
            "case_keys",
            ATTR_TYPE_I64_ARRAY,
            doc="Sorted unique selector values for explicit lookup cases.",
        ),
    ],
    traits=[PURE, SAFE_TO_SPECULATE],
    verify="loom_scf_lookup_verify",
    canonicalize="loom_scf_lookup_canonicalize",
    facts="loom_scf_lookup_facts",
    format=[
        Ref("selector"),
        AttrTable("case_keys", "values"),
        COLON,
        ResultTypeList("results", parens=False),
    ],
    examples=[
        "%ordinal, %wgx = scf.lookup %variant {0 = (%gemm0, %x0), 1 = (%gemm1, %x1)} default(%fallback, %xf) : index, index",
    ],
)

# ============================================================================
# scf.for — bounded counted loop
# ============================================================================
#
# Iterates an induction variable from lower_bound (inclusive) to
# upper_bound (exclusive) by step. Optionally carries loop state via
# a binding list that captures values into the body's entry block.
# Body terminated by scf.yield.
#
# Surface syntax pattern:
#
#   scf.for %iv = [%lo to %hi step %step](%a = %init : T) -> (T) { body }
#
# The bracketed range groups the iteration bounds. The parenthesized
# binding list (when present) captures values that the loop carries
# across iterations — these become block args on the body region's
# entry block alongside the induction variable. There is no
# `iter_args` keyword: the parens themselves are the syntactic marker
# and reading "for iv = range(captures) -> results" mirrors the
# function-call shape rather than tagging the captures with a name
# that only describes their role.

scf_for = Op(
    "scf.for",
    group=scf_ops,
    doc="Bounded counted loop with optional loop-carried state.",
    canonicalize="loom_scf_for_canonicalize",
    operands=[
        Operand("lower_bound", INDEX),
        Operand("upper_bound", INDEX),
        Operand("step", INDEX),
        Operand("iter_args", ANY, variadic=True),
    ],
    results=[Result("results", ANY, variadic=True)],
    regions=[
        RegionDef(
            "body",
            doc="Loop body. Terminated by scf.yield.",
            single_block=True,
            implicit_args=(("iv", "index"),),
        ),
    ],
    interfaces=[
        LoopLikeInterface(
            body="body",
            iter_args="iter_args",
            iv="iv",
            lower_bound="lower_bound",
            upper_bound="upper_bound",
            step="step",
        ),
    ],
    constraints=[
        IterArgsMatchResults("iter_args", "results"),
        YieldCountMatchesResults("body", "results"),
        YieldTypesMatchResults("body", "results"),
    ],
    format=[
        Ref("iv"),
        EQUALS,
        LBRACKET,
        Ref("lower_bound"),
        kw("to"),
        Ref("upper_bound"),
        kw("step"),
        Ref("step"),
        RBRACKET,
        OptionalGroup(
            [BindingList("iter_args")],
            anchor="iter_args",
        ),
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
        Region("body"),
    ],
    examples=[
        "scf.for %iv = [%c0 to %n step %c1] {\n  scf.yield\n}",
        "%result = scf.for %iv = [%c0 to %n step %c1](%acc = %init : f32) -> (f32) {\n  %next = scalar.addf %acc, %acc : f32\n  scf.yield %next : f32\n}",
    ],
)

# ============================================================================
# scf.if — conditional with required else
# ============================================================================
#
# Executes one of two regions based on a boolean condition. Both
# regions are required, regardless of whether the op produces
# results. Optional else (eliding the else region for void ifs)
# requires a RegionDef(optional=True) DSL extension and is deferred
# to a follow-up bead.

scf_if = Op(
    "scf.if",
    group=scf_ops,
    doc="Conditional execution with required else region.",
    canonicalize="loom_scf_if_canonicalize",
    operands=[Operand("condition", I1)],
    results=[Result("results", ANY, variadic=True)],
    regions=[
        RegionDef(
            "then_region",
            doc="Executed when condition is true. Terminated by scf.yield.",
            single_block=True,
        ),
        RegionDef(
            "else_region",
            doc="Executed when condition is false. Terminated by scf.yield.",
            single_block=True,
        ),
    ],
    interfaces=[
        RegionBranchInterface(selector="condition"),
    ],
    constraints=[
        YieldCountMatchesResults("then_region", "results"),
        YieldTypesMatchResults("then_region", "results"),
        YieldCountMatchesResults("else_region", "results"),
        YieldTypesMatchResults("else_region", "results"),
    ],
    format=[
        Ref("condition"),
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
        Region("then_region"),
        kw("else"),
        Region("else_region"),
    ],
    examples=[
        "scf.if %cond {\n  scf.yield\n} else {\n  scf.yield\n}",
        "%result = scf.if %cond -> (f32) {\n  scf.yield %a : f32\n} else {\n  scf.yield %b : f32\n}",
    ],
)

# ============================================================================
# All ops
# ============================================================================

ALL_SCF_OPS: tuple[Op, ...] = (
    scf_for,
    scf_if,
    scf_yield,
    scf_select,
    scf_lookup,
)
