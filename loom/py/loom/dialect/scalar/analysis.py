# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar analysis ops: constraints, assertions, and value properties.

These ops carry metadata about SSA values — they don't compute new
values, they assert or constrain properties of existing ones. The
optimizer and verifier use these constraints; codegen treats them as
identity (the output equals the input).
"""

from loom.assembly import COLON, PredicateList, Refs, TypesOf
from loom.dialect.scalar.defs import scalar_ops
from loom.dsl import ANY, PURE, AttrDef, Op, Operand, Result

__all__ = [
    "scalar_assume",
    "ALL_ANALYSIS_OPS",
]

# ============================================================================
# scalar.assume — predicate-constrained identity
# ============================================================================

scalar_assume = Op(
    "scalar.assume",
    group=scalar_ops,
    doc="Identity with predicate constraints on results.",
    operands=[Operand("values", ANY, variadic=True)],
    results=[Result("results", ANY, variadic=True)],
    attrs=[AttrDef("predicates", "predicate_list")],
    traits=[PURE],
    fold="loom_scalar_assume_fold",
    format=[
        Refs("values"),
        PredicateList("predicates"),
        COLON,
        TypesOf("results"),
    ],
    examples=[
        "%M2 = scalar.assume %M [mul(%M, 16)] : index",
        "%M2, %K2 = scalar.assume %M, %K [mul(%M, 16), lt(%K, 1024)] : index, index",
    ],
)

# ============================================================================
# Registry
# ============================================================================

ALL_ANALYSIS_OPS: tuple[Op, ...] = (scalar_assume,)
