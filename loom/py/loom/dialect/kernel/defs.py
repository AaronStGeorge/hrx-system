# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Kernel dialect op definitions."""

from loom.assembly import AttrDict
from loom.dsl import (
    ATTR_TYPE_ENUM,
    UNKNOWN_EFFECTS,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
    Op,
)

# ============================================================================
# Dialect
# ============================================================================

kernel_ops = Dialect(
    "kernel",
    dialect_id=0x10,
    doc="Kernel execution and synchronization operations.",
)

# ============================================================================
# Shared attrs
# ============================================================================

KernelScope = EnumDef(
    "KernelScope",
    [
        EnumCase("thread", 0, doc="Current invocation or thread."),
        EnumCase("subgroup", 1, doc="Current SIMD subgroup or wave."),
        EnumCase("workgroup", 2, doc="Current workgroup or block."),
        EnumCase("device", 3, doc="Current device."),
        EnumCase("system", 4, doc="Whole system."),
    ],
    doc="Target-independent synchronization scope.",
)

KernelMemorySpace = EnumDef(
    "KernelMemorySpace",
    [
        EnumCase("unknown", 0, doc="No target-independent memory space is known."),
        EnumCase("global", 1, doc="Device-visible global storage."),
        EnumCase("workgroup", 2, doc="Workgroup/shared storage."),
        EnumCase("private", 3, doc="Invocation-private storage."),
        EnumCase("constant", 4, doc="Read-only constant storage."),
        EnumCase("host", 5, doc="Host-visible storage."),
        EnumCase("descriptor", 6, doc="Descriptor-backed storage identity."),
    ],
    doc="Target-independent memory space fenced by a kernel synchronization op.",
)

KernelOrdering = EnumDef(
    "KernelOrdering",
    [
        EnumCase("relaxed", 0, doc="Atomicity without inter-address synchronization."),
        EnumCase("acquire", 1, doc="Acquire ordering."),
        EnumCase("release", 2, doc="Release ordering."),
        EnumCase("acq_rel", 3, doc="Acquire and release ordering."),
        EnumCase("seq_cst", 4, doc="Sequentially consistent ordering."),
    ],
    doc="Target-independent memory ordering for kernel synchronization ops.",
)

# ============================================================================
# kernel.barrier — workgroup execution barrier with an explicit memory fence
# ============================================================================

kernel_barrier = Op(
    name="kernel.barrier",
    group=kernel_ops,
    doc=(
        "Synchronize invocations in an explicit execution scope and fence a "
        "named memory space with a required ordering. The initial supported "
        "form is a workgroup barrier over workgroup memory with acquire-release "
        "ordering; other enum cases are reserved until their lowering and "
        "legality rules are implemented."
    ),
    attrs=[
        AttrDef(
            "memory_space",
            ATTR_TYPE_ENUM,
            enum_def=KernelMemorySpace,
            doc="Memory space whose accesses are fenced by the barrier.",
        ),
        AttrDef(
            "ordering",
            ATTR_TYPE_ENUM,
            enum_def=KernelOrdering,
            doc="Memory ordering applied to fenced accesses.",
        ),
        AttrDef(
            "scope",
            ATTR_TYPE_ENUM,
            enum_def=KernelScope,
            doc="Execution scope synchronized by the barrier.",
        ),
    ],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_kernel_barrier_verify",
    format=[AttrDict()],
    examples=[
        "kernel.barrier {memory_space = workgroup, ordering = acq_rel, scope = workgroup}",
    ],
)

# ============================================================================
# Registry
# ============================================================================

ALL_KERNEL_OPS: tuple[Op, ...] = (kernel_barrier,)
