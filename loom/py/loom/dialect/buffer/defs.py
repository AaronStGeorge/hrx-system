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
    Ref,
    ResultType,
    TypeOf,
)
from loom.dsl import (
    BUFFER,
    OFFSET,
    PURE,
    VIEW,
    Dialect,
    Op,
    Operand,
    Result,
)

# ============================================================================
# Group
# ============================================================================

buffer_ops = Dialect(
    "buffer",
    dialect_id=0x0C,
    doc="Opaque storage roots and typed view construction.",
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
    traits=[PURE],
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

ALL_BUFFER_OPS: tuple[Op, ...] = (buffer_view,)
