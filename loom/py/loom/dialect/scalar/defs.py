# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar dialect definitions: op group and flag enums.

Contains the scalar Dialect and its associated EnumDef types
(FastMathFlags, IntOverflowFlags) that are shared across all scalar
submodules.
"""

from loom.dsl import Dialect, EnumCase, EnumDef

__all__ = [
    "scalar_ops",
    "FastMathFlags",
    "IntOverflowFlags",
]

scalar_ops = Dialect("scalar", dialect_id=0x02, doc="Scalar arithmetic, math, and conversion ops.")

FastMathFlags = EnumDef(
    "FastMathFlags",
    [
        EnumCase("reassoc", 1, doc="Allow reassociation."),
        EnumCase("nnan", 2, doc="Assume no NaNs."),
        EnumCase("ninf", 4, doc="Assume no infinities."),
        EnumCase("nsz", 8, doc="Assume no signed zeros."),
        EnumCase("arcp", 16, doc="Allow reciprocal approximation."),
        EnumCase("contract", 32, doc="Allow contractions (FMA)."),
        EnumCase("afn", 64, doc="Approximate functions."),
        EnumCase("fast", 127, doc="All of the above."),
    ],
    doc="IEEE 754 fast-math relaxation flags for float operations.",
)

IntOverflowFlags = EnumDef(
    "IntOverflowFlags",
    [
        EnumCase("nsw", 1, doc="No signed wrap."),
        EnumCase("nuw", 2, doc="No unsigned wrap."),
    ],
    doc="Integer overflow behavior flags.",
)
