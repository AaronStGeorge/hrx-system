# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SCF dialect: structured control flow operations.

Provides counted loops (scf.for), conditionals (scf.if), and region
terminators (scf.yield). Used at every IR abstraction level — model
(layer loops, conditional architecture dispatch), tile (pipelined
K-reduction), and vector (unrolled inner loops).

Design doc: .notes/loom/scf-dialect.md
"""

from loom.dialect.scf.defs import (
    ALL_SCF_OPS,
    scf_for,
    scf_if,
    scf_ops,
    scf_yield,
)

__all__ = [
    "scf_ops",
    "scf_for",
    "scf_if",
    "scf_yield",
    "ALL_SCF_OPS",
]
