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
    ENCODING,
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
    doc="View layout construction and logical subview operations.",
)

# ============================================================================
# view.layout.dense — dense logical-to-physical layout
# ============================================================================

view_layout_dense = Op(
    name="view.layout.dense",
    group=view_ops,
    doc=("Construct a dense row-major address layout. The consuming view type provides the rank and logical extents."),
    results=[Result("result", ENCODING, doc="Dense address-layout value.")],
    traits=[PURE],
    format=[COLON, ResultType("result")],
    examples=[
        "%layout = view.layout.dense : encoding",
    ],
)

# ============================================================================
# view.layout.strided — explicit element-stride layout
# ============================================================================

view_layout_strided = Op(
    name="view.layout.strided",
    group=view_ops,
    doc=("Construct an address layout from per-dimension element strides. Static and dynamic stride values are interleaved in one bracket list."),
    operands=[Operand("strides", INDEX, doc="Dynamic element strides.", variadic=True)],
    results=[Result("result", ENCODING, doc="Strided address-layout value.")],
    attrs=[
        AttrDef(
            "static_strides",
            ATTR_TYPE_I64_ARRAY,
            doc="Static element strides with INT64_MIN sentinels for dynamics.",
        ),
    ],
    traits=[PURE],
    verify="loom_view_layout_strided_verify",
    format=[
        IndexList("strides", "static_strides"),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%layout = view.layout.strided [%row_stride, 1] : encoding",
        "%layout = view.layout.strided [4096, 1] : encoding",
    ],
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

ALL_VIEW_OPS: tuple[Op, ...] = (
    view_layout_dense,
    view_layout_strided,
    view_subview,
)
