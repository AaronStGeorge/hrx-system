# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Kernel dialect: kernel execution and synchronization operations."""

from loom.dialect.kernel.defs import (
    ALL_KERNEL_OPS,
    KernelMemorySpace,
    KernelOrdering,
    KernelScope,
    kernel_barrier,
    kernel_ops,
)

__all__ = [
    "kernel_ops",
    "ALL_KERNEL_OPS",
    "KernelMemorySpace",
    "KernelOrdering",
    "KernelScope",
    "kernel_barrier",
]
