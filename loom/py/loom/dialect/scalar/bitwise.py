# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar bitwise: bitwise ops, shifts, rotates, bit counting."""

from loom.dialect.scalar import IntOverflowFlags, scalar_ops
from loom.dsl import (
    INTEGER,
    Op,
    binary_op,
    unary_op,
)

__all__ = ["ALL_BITWISE_OPS"]

# ============================================================================
# Bitwise logic
# ============================================================================

scalar_andi = binary_op(
    "scalar.andi",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Bitwise AND.",
    commutative=True,
    fold="loom_scalar_andi_fold",
    examples=["%result = scalar.andi %lhs, %rhs : i32"],
)
scalar_ori = binary_op(
    "scalar.ori",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Bitwise OR.",
    commutative=True,
    fold="loom_scalar_ori_fold",
    examples=["%result = scalar.ori %lhs, %rhs : i32"],
)
scalar_xori = binary_op(
    "scalar.xori",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Bitwise XOR.",
    commutative=True,
    fold="loom_scalar_xori_fold",
    examples=["%result = scalar.xori %lhs, %rhs : i32"],
)

# ============================================================================
# Shifts (with optional overflow flags on left shift)
# ============================================================================

scalar_shli = binary_op(
    "scalar.shli",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Left shift.",
    flags=("overflow", IntOverflowFlags),
    fold="loom_scalar_shli_fold",
    examples=["%result = scalar.shli %lhs, %rhs : i32"],
)
scalar_shrsi = binary_op(
    "scalar.shrsi",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Arithmetic right shift (sign-extending).",
    fold="loom_scalar_shrsi_fold",
    examples=["%result = scalar.shrsi %lhs, %rhs : i32"],
)
scalar_shrui = binary_op(
    "scalar.shrui",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Logical right shift (zero-extending).",
    fold="loom_scalar_shrui_fold",
    examples=["%result = scalar.shrui %lhs, %rhs : i32"],
)

# ============================================================================
# Rotates
# ============================================================================

scalar_rotli = binary_op(
    "scalar.rotli",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Left rotate.",
    fold="loom_scalar_rotli_fold",
    examples=["%result = scalar.rotli %lhs, %rhs : i32"],
)
scalar_rotri = binary_op(
    "scalar.rotri",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Right rotate.",
    fold="loom_scalar_rotri_fold",
    examples=["%result = scalar.rotri %lhs, %rhs : i32"],
)

# ============================================================================
# Bit counting
# ============================================================================

scalar_ctlzi = unary_op(
    "scalar.ctlzi",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Count leading zeros.",
    fold="loom_scalar_ctlzi_fold",
    examples=["%result = scalar.ctlzi %input : i32"],
)
scalar_cttzi = unary_op(
    "scalar.cttzi",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Count trailing zeros.",
    fold="loom_scalar_cttzi_fold",
    examples=["%result = scalar.cttzi %input : i32"],
)
scalar_ctpopi = unary_op(
    "scalar.ctpopi",
    group=scalar_ops,
    type_constraint=INTEGER,
    doc="Population count (number of set bits).",
    fold="loom_scalar_ctpopi_fold",
    examples=["%result = scalar.ctpopi %input : i32"],
)

# ============================================================================
# Registry
# ============================================================================

ALL_BITWISE_OPS: tuple[Op, ...] = (
    scalar_andi,
    scalar_ori,
    scalar_xori,
    scalar_shli,
    scalar_shrsi,
    scalar_shrui,
    scalar_rotli,
    scalar_rotri,
    scalar_ctlzi,
    scalar_cttzi,
    scalar_ctpopi,
)
