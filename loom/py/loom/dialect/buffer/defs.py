# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Buffer dialect op definitions."""

from loom.assembly import (
    ARROW,
    COLON,
    GLUE,
    LBRACKET,
    RBRACKET,
    AttrDict,
    Ref,
    ResultType,
    TypeOf,
)
from loom.dsl import (
    ATTR_TYPE_ENUM,
    ATTR_TYPE_I64,
    BUFFER,
    OFFSET,
    PURE,
    REFINABLE_RESULT_TYPE_REFS,
    VIEW,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
    Op,
    Operand,
    OpPhase,
    Result,
)

# ============================================================================
# Group
# ============================================================================

buffer_ops = Dialect(
    "buffer",
    dialect_id=0x0C,
    doc="Opaque storage roots and typed view construction.",
    default_phase=OpPhase.EXECUTABLE,
)

# ============================================================================
# Shared attrs
# ============================================================================

MemorySpace = EnumDef(
    "MemorySpace",
    [
        EnumCase("unknown", 0, doc="No target-independent memory space is known."),
        EnumCase("global", 1, doc="Device-visible global storage."),
        EnumCase("workgroup", 2, doc="Workgroup/shared storage."),
        EnumCase("private", 3, doc="Invocation-private storage."),
        EnumCase("constant", 4, doc="Read-only constant storage."),
        EnumCase("host", 5, doc="Host-visible storage."),
        EnumCase("descriptor", 6, doc="Descriptor-backed storage identity."),
        EnumCase(
            "generic",
            7,
            doc=("Target-generic device storage. Targets may lower this to a generic address space such as AMDGPU flat memory."),
        ),
    ],
    doc="Target-independent memory space for buffer roots and derived views.",
)

# ============================================================================
# buffer.alloca — fixed-frame scratch allocation root
# ============================================================================

buffer_alloca = Op(
    name="buffer.alloca",
    group=buffer_ops,
    doc=(
        "Create a fixed-frame scratch buffer root in workgroup or private memory. "
        "Each execution produces a distinct storage identity; identical allocas "
        "must not be commoned. The byte length is a physical byte count, and "
        "base_alignment is the minimum byte alignment of the root storage base."
    ),
    operands=[
        Operand("byte_length", OFFSET, doc="Physical byte length of the scratch root."),
    ],
    results=[
        Result(
            "result",
            BUFFER,
            doc="Fresh opaque scratch storage root.",
            allocates=True,
        ),
    ],
    attrs=[
        AttrDef(
            "base_alignment",
            ATTR_TYPE_I64,
            doc="Minimum byte alignment of the root storage base.",
        ),
        AttrDef(
            "memory_space",
            ATTR_TYPE_ENUM,
            enum_def=MemorySpace,
            doc="Scratch memory space for the root; must be workgroup or private.",
        ),
    ],
    verify="loom_buffer_alloca_verify",
    facts="loom_buffer_alloca_facts",
    format=[
        Ref("byte_length"),
        AttrDict(),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%scratch = buffer.alloca %bytes {base_alignment = 64, memory_space = workgroup} : buffer",
    ],
)

# ============================================================================
# buffer.assume.memory_space — refine a buffer root memory-space fact
# ============================================================================

buffer_assume_memory_space = Op(
    name="buffer.assume.memory_space",
    group=buffer_ops,
    doc=("Refine an existing buffer root with a concrete target-independent memory-space fact while preserving the same storage identity, extent, alignment, and nullability facts."),
    operands=[Operand("buffer", BUFFER, doc="Buffer root to refine.")],
    results=[
        Result("result", BUFFER, doc="Same buffer root with refined memory-space facts."),
    ],
    attrs=[
        AttrDef(
            "memory_space",
            ATTR_TYPE_ENUM,
            enum_def=MemorySpace,
            doc="Concrete memory space to assume.",
        ),
    ],
    traits=[PURE],
    verify="loom_buffer_assume_memory_space_verify",
    facts="loom_buffer_assume_memory_space_facts",
    format=[
        Ref("buffer"),
        AttrDict(),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%global = buffer.assume.memory_space %buffer {memory_space = global} : buffer",
    ],
)

# ============================================================================
# buffer.view — form a typed view from a buffer root
# ============================================================================

buffer_view = Op(
    name="buffer.view",
    group=buffer_ops,
    doc=("Form a typed non-owning view from an opaque buffer root and base byte offset. The result view type carries the address layout."),
    operands=[
        Operand("buffer", BUFFER, doc="Opaque storage root."),
        Operand("byte_offset", OFFSET, doc="Base byte offset from the buffer root."),
    ],
    results=[Result("result", VIEW, doc="Typed logical view over the buffer.")],
    traits=[PURE, REFINABLE_RESULT_TYPE_REFS],
    verify="loom_buffer_view_verify",
    facts="loom_buffer_view_facts",
    format=[
        Ref("buffer"),
        GLUE,
        LBRACKET,
        Ref("byte_offset"),
        RBRACKET,
        COLON,
        TypeOf("buffer"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%view = buffer.view %buffer[%offset] : buffer -> view<[%M]xf32, %layout>",
    ],
)

# ============================================================================
# Registry
# ============================================================================

ALL_BUFFER_OPS: tuple[Op, ...] = (
    buffer_alloca,
    buffer_assume_memory_space,
    buffer_view,
)
