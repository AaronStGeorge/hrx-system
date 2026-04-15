# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLVM IR target dialect operations."""

from loom.dialect.llvmir.defs import (
    ALL_LLVMIR_OPS,
    AsmFlags,
    llvmir_inline_asm,
    llvmir_ops,
)

__all__ = [
    "llvmir_ops",
    "AsmFlags",
    "llvmir_inline_asm",
    "ALL_LLVMIR_OPS",
]
