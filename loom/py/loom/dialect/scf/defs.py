# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SCF dialect op definitions.

Three ops in this initial cut:

  scf.for     — Bounded counted loop with bracketed range syntax.
  scf.if      — Conditional execution with both regions required.
  scf.yield   — Variadic terminator for all scf op regions.

Always-explicit yield (no implicit terminator). Every scf op region
ends with a literal scf.yield, so passes walking region terminators
have a single uniform op kind to handle.

Design doc: .notes/loom/scf-dialect.md
"""

from loom.assembly import (
    ARROW,
    COLON,
    EQUALS,
    LBRACKET,
    RBRACKET,
    BindingList,
    OptionalGroup,
    Ref,
    Refs,
    Region,
    ResultTypeList,
    TypesOf,
    kw,
)
from loom.dsl import (
    ANY,
    I1,
    INDEX,
    TERMINATOR,
    Dialect,
    IterArgsMatchResults,
    LoopLikeInterface,
    Op,
    Operand,
    RegionBranchInterface,
    RegionDef,
    Result,
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
        LoopLikeInterface(body="body", iter_args="iter_args", iv="iv"),
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
)
