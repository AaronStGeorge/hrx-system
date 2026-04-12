# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""View dialect op definitions."""

from loom.assembly import (
    ARROW,
    COLON,
    IndexList,
    Ref,
    ResultType,
    TypeOf,
)
from loom.dsl import (
    ATTR_TYPE_I64_ARRAY,
    INDEX,
    PURE,
    VIEW,
    AttrDef,
    Dialect,
    Op,
    Operand,
    RanksMatch,
    Result,
    SameElementType,
    SameEncoding,
)

# ============================================================================
# Group
# ============================================================================

view_ops = Dialect(
    "view",
    dialect_id=0x0D,
    doc="Logical view operations.",
)

# ============================================================================
# view.subview — logical subview with explicit offsets
# ============================================================================

view_subview = Op(
    name="view.subview",
    group=view_ops,
    doc=("Form a logical subview from an existing view. Offsets select the logical origin; result type dimensions provide the subview extents."),
    operands=[
        Operand("source", VIEW, doc="Source view."),
        Operand("offsets", INDEX, doc="Dynamic logical offsets.", variadic=True),
    ],
    results=[Result("result", VIEW, doc="Subview over the same storage root.")],
    attrs=[
        AttrDef(
            "static_offsets",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical offsets with INT64_MIN sentinels for dynamics.",
        ),
    ],
    constraints=[
        SameElementType("source", "result"),
        SameEncoding("source", "result"),
        RanksMatch("source", "result"),
    ],
    traits=[PURE],
    verify="loom_view_subview_verify",
    format=[
        Ref("source"),
        IndexList("offsets", "static_offsets"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%sub = view.subview %source[%row, 0] : view<[%M]x[%N]xf32, %layout> -> view<16x[%N]xf32, %layout>",
    ],
)

# ============================================================================
# Registry
# ============================================================================

ALL_VIEW_OPS: tuple[Op, ...] = (view_subview,)
