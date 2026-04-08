# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test dialect op definitions.

These ops exist solely to validate the DSL, format elements, printer,
parser, and builder infrastructure. They are not part of the loom
compiler — they are the acceptance tests for the toolchain.

Every format element type appears in at least one test op. Every
DSL feature (variadic, optional, tied results, regions, enums,
constraints, traits) is exercised. If the test ops round-trip
through printer and parser, real dialect ops will too.
"""

from loom.assembly import (
    ARROW,
    BINDING_ELEMENT,
    COLON,
    COMMA,
    EQUALS,
    GLUE,
    LBRACKET,
    LPAREN,
    RBRACKET,
    RPAREN,
    Attr,
    AttrDict,
    BindingList,
    FuncArgs,
    IndexList,
    OptionalGroup,
    PredicateList,
    Ref,
    Refs,
    Region,
    ResultType,
    ResultTypeList,
    Scope,
    SymbolRef,
    TypeOf,
    TypesOf,
    kw,
)
from loom.dsl import (
    ANY,
    CONSTANT_LIKE,
    ELEMENTWISE,
    FLOAT,
    I1,
    INDEX,
    INTEGER,
    INVOLUTION,
    ISOLATED_FROM_ABOVE,
    POOL,
    PURE,
    SYMBOL_DEFINE,
    TENSOR,
    TERMINATOR,
    TILE,
    UNKNOWN_EFFECTS,
    AllShapesMatch,
    AttrDef,
    BlockArgCount,
    BlockArgsMatchElementTypes,
    Dialect,
    DimIndexInBounds,
    EnumCase,
    EnumDef,
    FuncLikeInterface,
    ImplicitTerminator,
    OffsetCountMatchesRank,
    Op,
    Operand,
    Reads,
    ReadWrites,
    RegionDef,
    Result,
    SameElementType,
    SameType,
    TiedResult,
    Writes,
    YieldCountMatchesResults,
    YieldElementTypesMatchResults,
    binary_op,
    cast_op,
    comparison_op,
    unary_op,
)

# ============================================================================
# Group and enums
# ============================================================================

test_ops = Dialect("test", dialect_id=0x01, doc="Test ops for infrastructure validation.")

# Enums for test.func (same values as the func dialect, defined
# locally to avoid cross-dialect import).
_Visibility = EnumDef(
    "Visibility",
    [EnumCase("public", 1, doc="Visible outside the module.")],
    doc="Function visibility. Absent (0) means private.",
)
_CallingConv = EnumDef(
    "CallingConv",
    [
        EnumCase("host", 1, doc="Host calling convention."),
        EnumCase("device", 2, doc="Device calling convention."),
        EnumCase("initializer", 3, doc="Module initialization."),
        EnumCase("deinitializer", 4, doc="Module deinitialization."),
    ],
    doc="Function calling convention. Absent (0) means host.",
)

cmp_predicates = EnumDef(
    "TestCmpPredicate",
    [
        EnumCase("eq", 0, doc="Equal."),
        EnumCase("ne", 1, doc="Not equal."),
        EnumCase("lt", 2, doc="Less than."),
        EnumCase("le", 3, doc="Less or equal."),
        EnumCase("gt", 4, doc="Greater than."),
        EnumCase("ge", 5, doc="Greater or equal."),
    ],
)

# ============================================================================
# test.addi — binary integer op
# ============================================================================

test_addi = binary_op(
    "test.addi",
    group=test_ops,
    type_constraint=INTEGER,
    doc="Test binary integer op.",
    commutative=True,
    fold="loom_test_addi_fold",
    canonicalize="loom_test_addi_canonicalize",
    examples=["%result = test.addi %lhs, %rhs : i32"],
)

# ============================================================================
# test.neg — unary float op
# ============================================================================

test_neg = unary_op(
    "test.neg",
    group=test_ops,
    type_constraint=FLOAT,
    doc="Test unary float op.",
    traits=[PURE, INVOLUTION],
    examples=["%result = test.neg %input : f32"],
)

# ============================================================================
# test.cast — type-casting op
# ============================================================================

test_cast = cast_op(
    "test.cast",
    group=test_ops,
    from_constraint=INTEGER,
    to_constraint=FLOAT,
    doc="Test cast op.",
    examples=["%result = test.cast %input : i32 to f32"],
)

# ============================================================================
# test.constant — constant materialization
# ============================================================================

test_constant = Op(
    "test.constant",
    group=test_ops,
    doc="Test constant materialization.",
    results=[Result("result", ANY)],
    attrs=[AttrDef("value", "any", doc="The constant value.")],
    traits=[PURE, CONSTANT_LIKE],
    fold="loom_test_constant_fold",
    verify="loom_test_constant_verify",
    format=[Attr("value"), COLON, TypeOf("result")],
    examples=[
        "%c42 = test.constant 42 : i32",
        "%pi = test.constant 3.14 : f32",
    ],
)

# ============================================================================
# test.use — side-effecting value sink for testing
# ============================================================================

test_use = Op(
    "test.use",
    group=test_ops,
    doc="Side-effecting sink that observes values without producing results. Not DCE-able. Use in tests to keep values alive for inspection.",
    operands=[Operand("values", ANY, variadic=True)],
    traits=[UNKNOWN_EFFECTS],
    format=[Refs("values"), COLON, TypesOf("values")],
    examples=[
        "test.use %a : i32",
        "test.use %a, %b : i32, f32",
    ],
)

# ============================================================================
# test.fact_* — value facts inspection ops for testing
# ============================================================================
#
# Each op reads one property from the analysis facts of its input and
# exposes it as a result. During canonicalization with facts enabled,
# the fold function returns exact facts, which the rewriter materializes
# as scalar.constant ops. This makes internal analysis state observable
# in .loom-test files.

test_fact_range_lo = Op(
    "test.fact_range_lo",
    group=test_ops,
    doc="Exposes the analysis range lower bound as an i64 constant.",
    operands=[Operand("value", ANY)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    fold="loom_test_fact_range_lo_fold",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%lo = test.fact_range_lo %x : index -> i64"],
)

test_fact_range_hi = Op(
    "test.fact_range_hi",
    group=test_ops,
    doc="Exposes the analysis range upper bound as an i64 constant.",
    operands=[Operand("value", ANY)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    fold="loom_test_fact_range_hi_fold",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%hi = test.fact_range_hi %x : index -> i64"],
)

test_fact_divisor = Op(
    "test.fact_divisor",
    group=test_ops,
    doc="Exposes the analysis known divisor as an i64 constant.",
    operands=[Operand("value", ANY)],
    results=[Result("result", INTEGER)],
    traits=[PURE],
    fold="loom_test_fact_divisor_fold",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%div = test.fact_divisor %x : index -> i64"],
)

test_fact_non_negative = Op(
    "test.fact_non_negative",
    group=test_ops,
    doc="Returns 1 if the input is provably non-negative, 0 otherwise.",
    operands=[Operand("value", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    fold="loom_test_fact_non_negative_fold",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%nn = test.fact_non_negative %x : index -> i1"],
)

test_fact_non_zero = Op(
    "test.fact_non_zero",
    group=test_ops,
    doc="Returns 1 if the input is provably non-zero, 0 otherwise.",
    operands=[Operand("value", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    fold="loom_test_fact_non_zero_fold",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%nz = test.fact_non_zero %x : index -> i1"],
)

test_fact_positive = Op(
    "test.fact_positive",
    group=test_ops,
    doc="Returns 1 if the input is provably positive (> 0), 0 otherwise.",
    operands=[Operand("value", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    fold="loom_test_fact_positive_fold",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%pos = test.fact_positive %x : index -> i1"],
)

test_fact_power_of_two = Op(
    "test.fact_power_of_two",
    group=test_ops,
    doc="Returns 1 if the input is provably a power of two, 0 otherwise.",
    operands=[Operand("value", ANY)],
    results=[Result("result", I1)],
    traits=[PURE],
    fold="loom_test_fact_power_of_two_fold",
    format=[Ref("value"), COLON, TypeOf("value"), ARROW, ResultType("result")],
    examples=["%p2 = test.fact_power_of_two %x : index -> i1"],
)

# ============================================================================
# test.cmp — comparison with enum predicate
# ============================================================================

test_cmp = comparison_op(
    "test.cmp",
    group=test_ops,
    type_constraint=INTEGER,
    predicates=cmp_predicates,
    doc="Test comparison op.",
    examples=["%result = test.cmp lt, %lhs, %rhs : i32"],
)

# ============================================================================
# test.map — region capture (elementwise-like)
# ============================================================================

test_map = Op(
    "test.map",
    group=test_ops,
    doc="Test region-capture elementwise op.",
    operands=[Operand("inputs", TILE, variadic=True)],
    results=[Result("result", TILE)],
    regions=[RegionDef("body", doc="Element-wise body.", single_block=True)],
    constraints=[
        AllShapesMatch("inputs"),
        BlockArgCount("body", "inputs"),
        BlockArgsMatchElementTypes("body", "inputs"),
        YieldCountMatchesResults("body", "result"),
        YieldElementTypesMatchResults("body", "result"),
    ],
    traits=[PURE, ELEMENTWISE, ImplicitTerminator("test.implicit_yield")],
    format=[
        BindingList("inputs", kind=BINDING_ELEMENT),
        Region("body"),
        ARROW,
        ResultTypeList("result"),
    ],
    examples=[
        "%result = test.map(%element = %input : tile<4xf32>) {\n  %negated = test.neg %element : f32\n  test.yield %negated : f32\n} -> (tile<4xf32>)",
    ],
)

# ============================================================================
# test.update — tied result with index list
# ============================================================================

test_update = Op(
    "test.update",
    group=test_ops,
    doc="Test tied result with index list.",
    operands=[
        Operand("source", TILE),
        Operand("target", TENSOR),
        Operand("offsets", INDEX, variadic=True),
    ],
    results=[TiedResult("result", "target", TENSOR)],
    attrs=[
        AttrDef(
            "static_offsets",
            "i64_array",
            doc="Static offset values (sentinel for dynamic).",
        ),
    ],
    constraints=[SameElementType("source", "target", "result")],
    traits=[PURE],
    format=[
        Ref("source"),
        COMMA,
        Ref("target"),
        IndexList("offsets", "static_offsets"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultTypeList("result"),
    ],
    examples=[
        "%result = test.update %tile, %tensor[%offset] : tile<4xf32> -> (%tensor as tensor<[%M]xf32>)",
    ],
)

# ============================================================================
# test.invoke — variadic call-like with tied results
# ============================================================================

test_invoke = Op(
    "test.invoke",
    group=test_ops,
    doc="Test variadic call-like op with tied results. The verifier checks that the invoke signature matches the referenced function declaration or definition.",
    operands=[
        Operand("operands", ANY, variadic=True),
    ],
    attrs=[
        AttrDef("callee", "symbol"),
    ],
    results=[Result("results", ANY, variadic=True)],
    traits=[UNKNOWN_EFFECTS],
    verify="loom_test_invoke_verify",
    format=[
        SymbolRef("callee"),
        GLUE,
        LPAREN,
        Refs("operands"),
        RPAREN,
        COLON,
        LPAREN,
        TypesOf("operands"),
        RPAREN,
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
    ],
    examples=[
        "%output, %count = test.invoke @callee(%weights, %input) : (tile<4xf32>, index) -> (%weights as tile<4xf32>, index)",
    ],
)

# ============================================================================
# test.slice — index list with mixed static/dynamic offsets
# ============================================================================

test_slice = Op(
    "test.slice",
    group=test_ops,
    doc="Test index list with mixed static/dynamic offsets.",
    operands=[
        Operand("source", TILE),
        Operand("offsets", INDEX, variadic=True),
    ],
    results=[Result("result", TILE)],
    attrs=[
        AttrDef("static_offsets", "i64_array", doc="Static offset values."),
    ],
    constraints=[SameElementType("source", "result"), OffsetCountMatchesRank("source", "offsets")],
    traits=[PURE],
    format=[
        Ref("source"),
        IndexList("offsets", "static_offsets"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultTypeList("result"),
    ],
    examples=[
        "%subtile = test.slice %source[0, %offset] : tile<64x64xf16> -> (tile<16x16xf16>)",
    ],
)

# ============================================================================
# test.loop — for-loop with iter_args and tied results
# ============================================================================

test_loop = Op(
    "test.loop",
    group=test_ops,
    doc="Test for-loop with iter_args and tied results.",
    operands=[
        Operand("lower_bound", INDEX),
        Operand("upper_bound", INDEX),
        Operand("step", INDEX),
        Operand("iter_args", ANY, variadic=True),
    ],
    results=[Result("results", ANY, variadic=True)],
    regions=[
        RegionDef(
            "body",
            doc="Loop body.",
            single_block=True,
            implicit_args=(("iv", "index"),),
        )
    ],
    traits=[ImplicitTerminator("test.implicit_yield")],
    format=[
        Ref("iv"),
        EQUALS,
        Ref("lower_bound"),
        kw("to"),
        Ref("upper_bound"),
        kw("step"),
        Ref("step"),
        OptionalGroup(
            [kw("iter_args"), BindingList("iter_args")],
            anchor="iter_args",
        ),
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
        Region("body"),
    ],
    examples=[
        "%result = test.loop %i = %c0 to %count step %c1 iter_args(%accumulator = %init : f32) -> (%init as f32) {\n  %next = test.neg %accumulator : f32\n  test.yield %next : f32\n}",
    ],
)

# ============================================================================
# test.branch — if/else with optional else region
# ============================================================================

test_branch = Op(
    "test.branch",
    group=test_ops,
    doc="Test if/else with both regions always present.",
    operands=[Operand("condition", INTEGER)],
    results=[Result("results", ANY, variadic=True)],
    regions=[
        RegionDef("then_region", doc="Then branch.", single_block=True),
        RegionDef("else_region", doc="Else branch.", single_block=True),
    ],
    traits=[ImplicitTerminator("test.implicit_yield")],
    format=[
        Ref("condition"),
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
        Region("then_region"),
        kw("else"),
        Region("else_region"),
    ],
    examples=[
        "%result = test.branch %condition -> (f32) {\n  test.yield %true_value : f32\n} else {\n  test.yield %false_value : f32\n}",
    ],
)

# ============================================================================
# test.implicit_yield — dedicated implicit region terminator
# ============================================================================

test_implicit_yield = Op(
    "test.implicit_yield",
    group=test_ops,
    doc="Dedicated zero-field implicit terminator synthesized for elidable test regions.",
    traits=[TERMINATOR],
    examples=["test.implicit_yield"],
)

# ============================================================================
# test.yield — variadic yield terminator
# ============================================================================

test_yield = Op(
    "test.yield",
    group=test_ops,
    doc="Test yield terminator.",
    operands=[Operand("values", ANY, variadic=True)],
    traits=[TERMINATOR],
    format=[
        OptionalGroup(
            [Refs("values"), COLON, TypesOf("values")],
            anchor="values",
        ),
    ],
    examples=["test.yield", "test.yield %first, %second : f32, i32"],
)

# ============================================================================
# test.func / test.decl — function definitions and declarations
# ============================================================================

test_func = Op(
    "test.func",
    group=test_ops,
    doc="Test function definition with body always present.",
    traits=[SYMBOL_DEFINE, ISOLATED_FROM_ABOVE],
    interfaces=[FuncLikeInterface(callee="callee", visibility="visibility", cc="cc", body="body")],
    attrs=[
        AttrDef("callee", "symbol"),
        AttrDef("visibility", "enum", enum_def=_Visibility, optional=True),
        AttrDef("cc", "enum", enum_def=_CallingConv, optional=True),
        AttrDef("predicates", "predicate_list", optional=True),
    ],
    results=[Result("results", ANY, variadic=True)],
    regions=[RegionDef("body", doc="Function body.")],
    format=[
        OptionalGroup([Attr("visibility")], anchor="visibility"),
        OptionalGroup([Attr("cc")], anchor="cc"),
        SymbolRef("callee"),
        Scope(
            [
                FuncArgs("args"),
                OptionalGroup(
                    [ARROW, ResultTypeList("results")],
                    anchor="results",
                ),
                OptionalGroup(
                    [kw("where"), PredicateList("predicates")],
                    anchor="predicates",
                ),
            ]
        ),
        Region("body"),
    ],
    examples=[
        "test.func @identity(%input: f32) -> (f32) {\n  test.yield %input : f32\n}",
    ],
)

test_decl = Op(
    "test.decl",
    group=test_ops,
    doc="Test function declaration with no body and signature arguments stored as op operands.",
    traits=[SYMBOL_DEFINE],
    interfaces=[
        FuncLikeInterface(
            callee="callee",
            visibility="visibility",
            cc="cc",
            args_as_operands=True,
        )
    ],
    attrs=[
        AttrDef("callee", "symbol"),
        AttrDef("visibility", "enum", enum_def=_Visibility, optional=True),
        AttrDef("cc", "enum", enum_def=_CallingConv, optional=True),
    ],
    results=[Result("results", ANY, variadic=True)],
    format=[
        OptionalGroup([Attr("visibility")], anchor="visibility"),
        OptionalGroup([Attr("cc")], anchor="cc"),
        SymbolRef("callee"),
        Scope(
            [
                FuncArgs("args"),
                OptionalGroup(
                    [ARROW, ResultTypeList("results")],
                    anchor="results",
                ),
            ]
        ),
    ],
    examples=[
        "test.decl @identity(%input: f32) -> (%input as f32)",
    ],
)

# ============================================================================
# test.attrs — op with attribute dictionary
# ============================================================================

test_attrs = Op(
    "test.attrs",
    group=test_ops,
    doc="Test op with attribute dictionary.",
    operands=[Operand("input", ANY)],
    results=[Result("result", ANY)],
    attrs=[AttrDef("dict", "dict", optional=True)],
    constraints=[SameType("input", "result")],
    traits=[PURE],
    format=[
        Ref("input"),
        AttrDict("dict"),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        '%result = test.attrs %input {axis = 0, label = "foo"} : f32',
    ],
)

# ============================================================================
# test.deflate — result dim references
# ============================================================================

test_deflate = Op(
    "test.deflate",
    group=test_ops,
    doc="Test op with result type referencing a co-result dim.",
    operands=[Operand("input", TENSOR)],
    results=[Result("results", ANY, variadic=True)],
    traits=[PURE],
    format=[
        Ref("input"),
        COLON,
        TypeOf("input"),
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
    ],
    examples=[
        "%output, %length = test.deflate %input : tensor<[%M]xf32> -> (tensor<[%length]xf32>, index)",
    ],
)

# ============================================================================
# test.assume — predicate-constrained identity
# ============================================================================

test_assume = Op(
    "test.assume",
    group=test_ops,
    doc="Test predicate-constrained identity (SSA assume).",
    operands=[Operand("values", ANY, variadic=True)],
    results=[Result("results", ANY, variadic=True)],
    attrs=[AttrDef("predicates", "predicate_list")],
    traits=[PURE],
    format=[
        Refs("values"),
        PredicateList("predicates"),
        COLON,
        TypesOf("results"),
    ],
    examples=[
        "%M2 = test.assume %M [mul(%M, 16)] : index",
        "%M2, %K2 = test.assume %M, %K [mul(%M, 16), lt(%K, 1024)] : index, index",
    ],
)

# ============================================================================
# test.convert — single bare result type (ResultType, no parens)
# ============================================================================

test_convert = Op(
    "test.convert",
    group=test_ops,
    doc="Test op with bare result type (no parentheses).",
    operands=[Operand("input", ANY)],
    results=[Result("result", ANY)],
    traits=[PURE],
    format=[
        Ref("input"),
        COLON,
        TypeOf("input"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%result = test.convert %input : i32 -> f32",
        "%tile = test.convert %x : tile<4xi8> -> tile<4xf32>",
    ],
)

# ============================================================================
# test.reduce — variadic SameType constraint
# ============================================================================

test_reduce = Op(
    "test.reduce",
    group=test_ops,
    doc="Test variadic operands with SameType constraint across variadic and result.",
    operands=[Operand("inputs", ANY, variadic=True)],
    results=[Result("result", ANY)],
    constraints=[SameType("inputs", "result")],
    traits=[PURE],
    format=[
        Refs("inputs"),
        COLON,
        TypeOf("result"),
    ],
    examples=[
        "%sum = test.reduce %a, %b, %c : i32",
    ],
)

# ============================================================================
# test.read_resource — read-only resource op (for effect testing)
# ============================================================================

test_read_resource = Op(
    "test.read_resource",
    group=test_ops,
    doc="Test op that reads from a resource operand.",
    operands=[Operand("source", POOL, doc="Resource to read from.")],
    results=[Result("result", ANY, doc="Data read from the resource.")],
    effects=[Reads("source")],
    format=[
        Ref("source"),
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%tile = test.read_resource %pool : pool<[%BS]> -> tile<4xf32>",
    ],
)

# ============================================================================
# test.write_resource — write-only resource op (for effect testing)
# ============================================================================

test_write_resource = Op(
    "test.write_resource",
    group=test_ops,
    doc="Test op that writes to a resource operand.",
    operands=[
        Operand("target", POOL, doc="Resource to write to."),
        Operand("data", ANY, doc="Data to write."),
    ],
    effects=[Writes("target")],
    format=[
        Ref("target"),
        COMMA,
        Ref("data"),
        COLON,
        TypeOf("target"),
        COMMA,
        TypeOf("data"),
    ],
    examples=[
        "test.write_resource %pool, %tile : pool<[%BS]>, tile<4xf32>",
    ],
)

# ============================================================================
# test.mutate_resource — atomic read-modify-write on a resource (for effect testing)
# ============================================================================

test_mutate_resource = Op(
    "test.mutate_resource",
    group=test_ops,
    doc="Test op that atomically reads and writes a resource operand.",
    operands=[
        Operand("target", POOL, doc="Resource to read-modify-write."),
        Operand("value", ANY, doc="Value to combine with the resource."),
    ],
    results=[Result("old_value", ANY, doc="Previous value from the resource.")],
    effects=[ReadWrites("target")],
    format=[
        Ref("target"),
        COMMA,
        Ref("value"),
        COLON,
        TypeOf("target"),
        COMMA,
        TypeOf("value"),
        ARROW,
        ResultType("old_value"),
    ],
    examples=[
        "%old = test.mutate_resource %pool, %delta : pool<[%BS]>, i32 -> i32",
    ],
)

# ============================================================================
# test.alloc — allocation op with unique identity (for CSE/DCE testing)
# ============================================================================

test_alloc = Op(
    "test.alloc",
    group=test_ops,
    doc="Test allocation op. Each execution produces a distinct identity even with identical operands. Prevents CSE but allows DCE when unused.",
    operands=[Operand("size", INDEX, doc="Allocation size.")],
    results=[Result("result", POOL, doc="Allocated pool.", allocates=True)],
    format=[
        Ref("size"),
        COLON,
        TypeOf("size"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%pool = test.alloc %sz : index -> pool<[%BS]>",
    ],
)

# ============================================================================
# test.isolated_region — isolated single-block region (for CSE testing)
# ============================================================================

test_isolated_region = Op(
    "test.isolated_region",
    group=test_ops,
    doc="Test op with an isolated single-block region. Values from the enclosing scope are not visible inside the body.",
    results=[Result("results", ANY, variadic=True)],
    regions=[
        RegionDef("body", doc="Isolated region body.", single_block=True),
    ],
    traits=[ISOLATED_FROM_ABOVE],
    format=[
        OptionalGroup(
            [ARROW, ResultTypeList("results")],
            anchor="results",
        ),
        Region("body"),
    ],
    examples=[
        "%r = test.isolated_region -> (i32) {\n  %c = test.constant 42 : i32\n  test.yield %c : i32\n}",
    ],
)

# ============================================================================
# test.counter — canonicalize test harness (multi-step, error, fixed point)
# ============================================================================

test_counter = Op(
    "test.counter",
    group=test_ops,
    doc="Test op for canonicalize multi-step and error path testing. Canonicalize: value < 0 returns error, value > 0 decrements, value == 0 is fixed point.",
    results=[Result("result", ANY)],
    attrs=[AttrDef("value", "i64", doc="Counter value.")],
    traits=[PURE],
    canonicalize="loom_test_counter_canonicalize",
    format=[Attr("value"), COLON, TypeOf("result")],
    examples=[
        "%c = test.counter 3 : i32",
        "%c = test.counter 0 : i32",
    ],
)

# ============================================================================
# test.dim — dimension query with ATTR_IN_RANGE_RANK constraint
# ============================================================================

test_dim = Op(
    "test.dim",
    group=test_ops,
    doc="Test dimension query to exercise ATTR_IN_RANGE_RANK constraint.",
    operands=[Operand("source", TILE)],
    results=[Result("result", INDEX)],
    attrs=[
        AttrDef("dim_index", "i64", doc="Dimension index to query."),
    ],
    constraints=[DimIndexInBounds("source", "dim_index")],
    traits=[PURE],
    format=[
        Ref("source"),
        LBRACKET,
        Attr("dim_index"),
        RBRACKET,
        COLON,
        TypeOf("source"),
        ARROW,
        ResultType("result"),
    ],
    examples=[
        "%d = test.dim %t[0] : tile<4xf32> -> index",
    ],
)

# ============================================================================
# Registry: all test ops in declaration order
# ============================================================================

ALL_TEST_OPS: tuple[Op, ...] = (
    test_addi,
    test_neg,
    test_cast,
    test_constant,
    test_use,
    test_cmp,
    test_map,
    test_update,
    test_invoke,
    test_slice,
    test_loop,
    test_branch,
    test_implicit_yield,
    test_yield,
    test_func,
    test_decl,
    test_attrs,
    test_deflate,
    test_assume,
    test_convert,
    test_reduce,
    test_read_resource,
    test_write_resource,
    test_mutate_resource,
    test_alloc,
    test_isolated_region,
    test_counter,
    test_dim,
    test_fact_range_lo,
    test_fact_range_hi,
    test_fact_divisor,
    test_fact_non_negative,
    test_fact_non_zero,
    test_fact_positive,
    test_fact_power_of_two,
)
