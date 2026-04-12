# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Encoding dialect op definitions."""

from loom.assembly import (
    COLON,
    COMMA,
    Attr,
    IndexList,
    OperandDict,
    Ref,
    ResultType,
    TypeOf,
)
from loom.dsl import (
    ANY,
    ATTR_TYPE_DICT,
    ATTR_TYPE_ENCODING,
    ATTR_TYPE_I64_ARRAY,
    ATTR_TYPE_STRING,
    ENCODING,
    INDEX,
    INTEGER,
    PURE,
    AttrDef,
    Dialect,
    Op,
    Operand,
    Result,
)

# ============================================================================
# Group
# ============================================================================

encoding_ops = Dialect("encoding", dialect_id=0x09, doc="Encoding definition and query ops.")

# ============================================================================
# encoding.layout.dense — dense logical-to-physical address layout
# ============================================================================

encoding_layout_dense = Op(
    name="encoding.layout.dense",
    group=encoding_ops,
    doc=("Construct a dense row-major address layout. The consuming view type provides the rank and logical extents."),
    results=[Result("result", ENCODING, doc="Dense address-layout value.")],
    traits=[PURE],
    format=[COLON, ResultType("result")],
    examples=[
        "%layout = encoding.layout.dense : encoding",
    ],
)

# ============================================================================
# encoding.layout.strided — explicit element-stride address layout
# ============================================================================

encoding_layout_strided = Op(
    name="encoding.layout.strided",
    group=encoding_ops,
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
    verify="loom_encoding_layout_strided_verify",
    format=[
        IndexList("strides", "static_strides"),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%layout = encoding.layout.strided [%row_stride, 1] : encoding",
        "%layout = encoding.layout.strided [4096, 1] : encoding",
    ],
)

# ============================================================================
# encoding.define — create an encoding value from a static specification
# ============================================================================

encoding_define = Op(
    name="encoding.define",
    group=encoding_ops,
    doc="Create an encoding value from a static encoding specification.",
    operands=[
        Operand("params", ANY, variadic=True),
    ],
    results=[Result("result", ENCODING)],
    attrs=[
        AttrDef("spec", ATTR_TYPE_ENCODING, doc="Static encoding specification."),
        AttrDef(
            "param_names",
            ATTR_TYPE_DICT,
            optional=True,
            doc="Sorted dynamic parameter names mapped to operand ordinals.",
        ),
    ],
    traits=[PURE],
    verify="loom_encoding_define_verify",
    format=[
        Attr("spec"),
        OperandDict("params", "param_names"),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%enc = encoding.define #q8_0<block=32> : encoding",
        "%enc = encoding.define #q8_0<block=32> {group_size = %group_size : index} : encoding",
    ],
)

# ============================================================================
# encoding.isa — test if an encoding belongs to a category
# ============================================================================

encoding_isa = Op(
    name="encoding.isa",
    group=encoding_ops,
    doc="Test if an encoding belongs to a category.",
    operands=[Operand("enc", ENCODING)],
    results=[Result("result", INTEGER)],
    attrs=[AttrDef("category", ATTR_TYPE_STRING)],
    traits=[PURE],
    format=[Ref("enc"), COMMA, Attr("category"), COLON, TypeOf("result")],
    examples=[
        '%is_quantized = encoding.isa %enc, "quantized" : i1',
    ],
)

# ============================================================================
# Registry: all encoding ops in declaration order
# ============================================================================

ALL_ENCODING_OPS: tuple[Op, ...] = (
    encoding_layout_dense,
    encoding_layout_strided,
    encoding_define,
    encoding_isa,
)
