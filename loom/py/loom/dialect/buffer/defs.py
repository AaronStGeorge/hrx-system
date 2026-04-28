# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Buffer dialect op definitions."""

from loom.assembly import (
    ARROW,
    COLON,
    COMMA,
    GLUE,
    LBRACKET,
    RBRACKET,
    AttrDict,
    Ref,
    ResultType,
    TypeOf,
)
from loom.dialect.memory import MemorySpace
from loom.dsl import (
    ATTR_TYPE_ENUM,
    ATTR_TYPE_I64,
    BUFFER,
    FACT_IDENTITY,
    OFFSET,
    PURE,
    REFINABLE_RESULT_TYPE_REFS,
    VIEW,
    AttrDef,
    Dialect,
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
    traits=[PURE, FACT_IDENTITY],
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
# buffer.assume.noalias — refine a buffer root with a comparable noalias scope
# ============================================================================

buffer_assume_noalias = Op(
    name="buffer.assume.noalias",
    group=buffer_ops,
    doc=(
        "Refine an existing buffer root with an explicit noalias contract. "
        "The result preserves the same storage identity, extent, memory-space, "
        "alignment, and nullability facts, and marks the root identity as "
        "comparable for disjointness proofs. External buffer arguments do not "
        "gain this proof by default."
    ),
    operands=[Operand("buffer", BUFFER, doc="Buffer root to refine.")],
    results=[Result("result", BUFFER, doc="Same buffer root with a noalias scope.")],
    traits=[PURE, FACT_IDENTITY],
    facts="loom_buffer_assume_noalias_facts",
    format=[
        Ref("buffer"),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%unique = buffer.assume.noalias %buffer : buffer",
    ],
)

# ============================================================================
# buffer.assume.same_root — refine a buffer root to share another root identity
# ============================================================================

buffer_assume_same_root = Op(
    name="buffer.assume.same_root",
    group=buffer_ops,
    doc=(
        "Refine an existing buffer root to share another buffer's storage root. "
        "This is a dominance-scoped assertion for internally specialized "
        "dispatches that know two incoming handles refer to the same allocation. "
        "The result keeps the first operand's value while inheriting the second "
        "operand's root identity and comparable alias scope."
    ),
    operands=[
        Operand("buffer", BUFFER, doc="Buffer value to refine."),
        Operand("root", BUFFER, doc="Buffer whose storage root is shared."),
    ],
    results=[Result("result", BUFFER, doc="Same buffer value with refined root identity.")],
    traits=[PURE],
    facts="loom_buffer_assume_same_root_facts",
    format=[
        Ref("buffer"),
        COMMA,
        Ref("root"),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%same = buffer.assume.same_root %buffer, %root : buffer",
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
    buffer_assume_noalias,
    buffer_assume_same_root,
    buffer_view,
)
