# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Global dialect op definitions.

Four ops for module-level state:

Top-level (module-level symbols):
  global.constant  — Immutable global (weights, parameters, constants).
  global.variable  — Mutable global (KV cache, running state).

Body ops (inside function/template bodies):
  global.load      — Load value + dynamic dims/encoding from global.
  global.store     — Store value + dynamic dims/encoding to global.

Globals are always module-private. Cross-module global references
would require a global.def/global.decl split (like func.def/func.decl)
with import semantics — future work.

Design document: .notes/loom/globals.md
"""

from loom.assembly import (
    COLON,
    COMMA,
    OptionalGroup,
    PredicateList,
    Ref,
    ResultType,
    Scope,
    SymbolRef,
    TypeOf,
    kw,
)
from loom.dsl import (
    ANY,
    SYMBOL_DEFINE,
    UNKNOWN_EFFECTS,
    AttrDef,
    Dialect,
    Op,
    Operand,
    Result,
)

# ============================================================================
# Op group
# ============================================================================

global_ops = Dialect(
    "global",
    dialect_id=0x0B,
    doc="Module-level state: constants, parameters, mutable state.",
)

# ============================================================================
# global.constant — immutable global value
# ============================================================================

global_constant = Op(
    "global.constant",
    group=global_ops,
    doc=(
        "Immutable global value. May have an inline initial value, a resource "
        "reference, or be initialized by an initializer function. After "
        "initialization, never written. Named type variables in the type "
        "annotation express structural constraints. Predicates constrain "
        "dynamic dimensions and are propagated to every load site as value "
        "facts."
    ),
    traits=[SYMBOL_DEFINE],
    attrs=[
        AttrDef("callee", "symbol"),
        AttrDef("predicates", "predicate_list", optional=True),
    ],
    # Single result carries the global's type. SYMBOL_DEFINE suppresses
    # the %name = prefix, so no SSA value is printed — just the type
    # after the colon.
    results=[Result("type", ANY)],
    format=[
        SymbolRef("callee"),
        COLON,
        Scope(
            [
                ResultType("type"),
                OptionalGroup(
                    [kw("where"), PredicateList("predicates")],
                    anchor="predicates",
                ),
            ]
        ),
    ],
    examples=[
        "global.constant @pi : f32",
        "global.constant @weights : tile<[%m]x[%k]xf32> where [mul(%m, 16)]",
        "global.constant @derived : tile<16x32xf32>",
    ],
)

# ============================================================================
# global.variable — mutable global value
# ============================================================================

global_variable = Op(
    "global.variable",
    group=global_ops,
    doc=("Mutable global value. Can be stored from any function at any time. Named type variables and predicates work the same as global.constant."),
    traits=[SYMBOL_DEFINE],
    attrs=[
        AttrDef("callee", "symbol"),
        AttrDef("predicates", "predicate_list", optional=True),
    ],
    results=[Result("type", ANY)],
    format=[
        SymbolRef("callee"),
        COLON,
        Scope(
            [
                ResultType("type"),
                OptionalGroup(
                    [kw("where"), PredicateList("predicates")],
                    anchor="predicates",
                ),
            ]
        ),
    ],
    examples=[
        "global.variable @kv_cache : tile<[%s]x[%d]xf32> where [mul(%s, 64)]",
        "global.variable @step_count : index",
    ],
)

# ============================================================================
# global.load — load value + dynamic dims/encoding from global
# ============================================================================

global_load = Op(
    "global.load",
    group=global_ops,
    doc=(
        "Load a value from a global. The type annotation uses #N result "
        "ordinals for fresh dynamic dims (new SSA values) and %name "
        "references for bound dims (existing SSA values, assertion). "
        "Predicates on the global definition are propagated as value facts."
    ),
    attrs=[
        AttrDef("global", "symbol"),
    ],
    # Variadic results: the loaded value + any fresh dim/encoding values.
    # The actual result count is determined by the type annotation at parse
    # time (one result per #N ordinal, plus the value itself).
    results=[Result("result", ANY, variadic=True)],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_global_load_verify",
    format=[
        SymbolRef("global"),
        COLON,
        ResultType("result"),
    ],
    examples=[
        "%tile, %m, %k = global.load @weights : tile<[#1]x[#2]xf32>",
        "%tile = global.load @bias : tile<[%m]xf32>",
        "%cache, %s, %d = global.load @kv_cache : tile<[#1]x[#1]x[#2]xf32>",
    ],
)

# ============================================================================
# global.store — store value + dynamic dims/encoding to global
# ============================================================================

global_store = Op(
    "global.store",
    group=global_ops,
    doc=(
        "Store a value to a global. Dynamic dims and encodings are captured "
        "implicitly from the type annotation, which references existing SSA "
        "values. The verifier checks that stores to global.constant only "
        "happen from initializer-reachable code paths."
    ),
    operands=[
        Operand("value", ANY, doc="The value to store."),
    ],
    attrs=[
        AttrDef("global", "symbol"),
    ],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_global_store_verify",
    format=[
        Ref("value"),
        COMMA,
        SymbolRef("global"),
        COLON,
        TypeOf("value"),
    ],
    examples=[
        "global.store %tile, @kv_cache : tile<[%m]xf32>",
        "global.store %table, @lut : tile<16x32xf32>",
    ],
)

# ============================================================================
# All ops
# ============================================================================

ALL_GLOBAL_OPS: tuple[Op, ...] = (
    global_constant,
    global_variable,
    global_load,
    global_store,
)
