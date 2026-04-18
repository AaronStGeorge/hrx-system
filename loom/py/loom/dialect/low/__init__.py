# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target-low structural dialect."""

from loom.dialect.low.defs import (
    ALL_LOW_OPS,
    low_abi_adapter,
    low_abi_operand,
    low_abi_result,
    low_const,
    low_copy,
    low_func_decl,
    low_func_def,
    low_invoke,
    low_op,
    low_ops,
    low_return,
)

__all__ = [
    "ALL_LOW_OPS",
    "low_const",
    "low_copy",
    "low_func_decl",
    "low_func_def",
    "low_abi_adapter",
    "low_abi_operand",
    "low_abi_result",
    "low_invoke",
    "low_op",
    "low_ops",
    "low_return",
]
