# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar dialect: primitive operations on scalar values.

Replaces MLIR's arith + math dialects with a single unified namespace.
These ops live inside tile.elementwise bodies, scf.for bounds, and
index computations. Exhaustively defined — every scalar operation the
compiler needs is declared here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# defs MUST be imported before submodules — submodules import
# scalar_ops/FastMathFlags/IntOverflowFlags from this package.
from loom.dialect.scalar.defs import (  # isort: split
    FastMathFlags,
    IntOverflowFlags,
    scalar_ops,
)

from loom.dialect.scalar.analysis import ALL_ANALYSIS_OPS
from loom.dialect.scalar.arithmetic import ALL_ARITHMETIC_OPS
from loom.dialect.scalar.bitwise import ALL_BITWISE_OPS
from loom.dialect.scalar.comparison import ALL_COMPARISON_OPS
from loom.dialect.scalar.conversion import ALL_CONVERSION_OPS
from loom.dialect.scalar.math import ALL_MATH_OPS

if TYPE_CHECKING:
    from loom.dsl import Op

__all__ = [
    "scalar_ops",
    "FastMathFlags",
    "IntOverflowFlags",
    "ALL_SCALAR_OPS",
]

ALL_SCALAR_OPS: tuple[Op, ...] = (
    *ALL_ARITHMETIC_OPS,
    *ALL_MATH_OPS,
    *ALL_COMPARISON_OPS,
    *ALL_CONVERSION_OPS,
    *ALL_BITWISE_OPS,
    *ALL_ANALYSIS_OPS,
)
