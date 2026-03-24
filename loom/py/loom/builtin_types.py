# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Built-in loom type declarations.

These TypeDefs define the textual format for the core loom types.
Scalar types (f32, i32, index) are keywords, not TypeDefs. Everything
with angle-bracket syntax (tile<...>, tensor<...>, group<...>) or
dotted-name syntax (hal.buffer, vm.ref<...>) is a TypeDef.

Dialect-specific types are declared in their respective dialect files
(e.g., dialect/hal/, dialect/vm/) using the same TypeDef pattern.
"""

from loom.assembly import (
    COMMA,
    Attr,
    EncodingOf,
    OptionalGroup,
    ScalarOf,
    ShapeOf,
    kw,
)
from loom.dsl import (
    ANY,
    EncodingParam,
    ScalarParam,
    ShapeParam,
    TypeDef,
    TypeParam,
)

__all__ = [
    "ALL_BUILTIN_TYPES",
    # Shaped types.
    "tile_type",
    "tensor_type",
    # Pool type.
    "pool_type",
    # Group type.
    "group_type",
]

# ============================================================================
# tile<...> — local compute tile
# ============================================================================

tile_type = TypeDef(
    name="tile",
    doc="Local compute tile with static/dynamic dims and optional encoding.",
    ir_kind="tile",
    params=[
        ShapeParam("dims"),
        ScalarParam("element_type"),
        EncodingParam("encoding"),
    ],
    format=[
        ShapeOf("dims"),
        kw("x"),
        ScalarOf("element_type"),
        OptionalGroup([COMMA, EncodingOf("encoding")], anchor="encoding"),
    ],
)

# ============================================================================
# tensor<...> — global storage tensor
# ============================================================================

tensor_type = TypeDef(
    name="tensor",
    doc="Global storage tensor with static/dynamic dims and optional encoding.",
    ir_kind="tensor",
    params=[
        ShapeParam("dims"),
        ScalarParam("element_type"),
        EncodingParam("encoding"),
    ],
    format=[
        ShapeOf("dims"),
        kw("x"),
        ScalarOf("element_type"),
        OptionalGroup([COMMA, EncodingOf("encoding")], anchor="encoding"),
    ],
)

# ============================================================================
# pool<...> — block-managed device memory pool
# ============================================================================

pool_type = TypeDef(
    name="pool",
    doc="Block-managed device memory pool with a single block size dimension.",
    ir_kind="pool",
    params=[
        ShapeParam("block_size"),
    ],
    format=[
        ShapeOf("block_size"),
    ],
)

# ============================================================================
# group<scope> — barrier scoping
# ============================================================================

group_type = TypeDef(
    name="group",
    doc="Barrier scoping type.",
    ir_kind="group",
    params=[TypeParam("scope", ANY)],
    format=[Attr("scope")],
)

# ============================================================================
# Registry
# ============================================================================

ALL_BUILTIN_TYPES: tuple[TypeDef, ...] = (
    tile_type,
    tensor_type,
    pool_type,
    group_type,
)
