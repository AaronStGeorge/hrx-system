# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""View dialect op definitions."""

from loom.assembly import (
    ARROW,
    COLON,
    AttrDict,
    IndexList,
    Ref,
    ResultType,
    TypeOf,
)
from loom.dsl import (
    ATTR_TYPE_ENUM,
    ATTR_TYPE_I64_ARRAY,
    HINT,
    INDEX,
    PURE,
    VIEW,
    AttrDef,
    Dialect,
    EnumCase,
    EnumDef,
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

PrefetchIntent = EnumDef(
    "PrefetchIntent",
    [
        EnumCase("read", 0, doc="Prefetch for an expected future read."),
        EnumCase("write", 1, doc="Prefetch for an expected future write."),
    ],
    doc="Intended future access kind for a prefetch hint.",
)

PrefetchLocality = EnumDef(
    "PrefetchLocality",
    [
        EnumCase("none", 0, doc="No temporal locality expected."),
        EnumCase("l1", 1, doc="Prefer L1 or the nearest target cache."),
        EnumCase("l2", 2, doc="Prefer L2 or the nearest mid-level target cache."),
        EnumCase("l3", 3, doc="Prefer L3 or the nearest outer target cache."),
    ],
    doc="Target-independent prefetch locality hint.",
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
    facts="loom_view_subview_facts",
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
# view.refine — type/fact refinement for an existing view
# ============================================================================

view_refine = Op(
    name="view.refine",
    group=view_ops,
    doc=(
        "Refine the static type information attached to an existing view while "
        "preserving the same storage root and byte base. This is an explicit "
        "SSA assertion point for layout, shape, and encoding facts discovered "
        "or required by earlier analysis."
    ),
    operands=[Operand("source", VIEW, doc="Source view to refine.")],
    results=[Result("result", VIEW, doc="Same view with refined type information.")],
    constraints=[
        SameElementType("source", "result"),
        RanksMatch("source", "result"),
    ],
    traits=[PURE],
    verify="loom_view_refine_verify",
    facts="loom_view_refine_facts",
    format=[
        Ref("source"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%refined = view.refine %view : view<[%M]xf32, %layout> -> view<16xf32, #dense>",
    ],
)

# ============================================================================
# view.prefetch — discardable compiler hint for a future view access
# ============================================================================

view_prefetch = Op(
    name="view.prefetch",
    group=view_ops,
    doc=(
        "Compiler hint for a future access to a logical view origin. Prefetch "
        "has no semantic memory effects and may not fault semantically, but it "
        "is intentionally preserved by ordinary canonicalization/DCE until an "
        "explicit hint-stripping pass removes it."
    ),
    operands=[
        Operand("view", VIEW, doc="Typed view whose address should be prefetched."),
        Operand("indices", INDEX, doc="Dynamic logical origin indices.", variadic=True),
    ],
    attrs=[
        AttrDef("intent", ATTR_TYPE_ENUM, enum_def=PrefetchIntent, doc="Required expected future access kind."),
        AttrDef("locality", ATTR_TYPE_ENUM, enum_def=PrefetchLocality, doc="Required target-independent locality hint."),
        AttrDef(
            "static_indices",
            ATTR_TYPE_I64_ARRAY,
            doc="Static logical origin indices with INT64_MIN sentinels for dynamics.",
        ),
    ],
    traits=[HINT],
    verify="loom_view_prefetch_verify",
    format=[
        Ref("view"),
        IndexList("indices", "static_indices"),
        AttrDict(),
        COLON,
        TypeOf("view"),
    ],
    examples=[
        "view.prefetch %view[%row, %col] {intent = read, locality = l2} : view<[%M]x[%N]xf32, %layout>",
    ],
)

# ============================================================================
# Registry
# ============================================================================

ALL_VIEW_OPS: tuple[Op, ...] = (view_subview, view_prefetch, view_refine)
