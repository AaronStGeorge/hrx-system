# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SCF dialect: structured control flow operations.

Provides counted loops (scf.for), conditionals (scf.if), whole-value
selection (scf.select), and region terminators (scf.yield). Used at every IR
abstraction level: model (layer loops, conditional architecture dispatch), tile
(pipelined K-reduction), and vector (unrolled inner loops).
"""

from loom.dialect.scf.defs import (
    ALL_SCF_OPS,
    scf_for,
    scf_if,
    scf_lookup,
    scf_ops,
    scf_select,
    scf_yield,
)

__all__ = [
    "scf_ops",
    "scf_for",
    "scf_if",
    "scf_lookup",
    "scf_select",
    "scf_yield",
    "ALL_SCF_OPS",
]
