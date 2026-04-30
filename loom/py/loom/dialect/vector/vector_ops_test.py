# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

from loom.assembly import OperandDict, OptionalGroup, PredicateList, ResultType
from loom.dialect.cache import CacheScope, CacheTemporal
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.vector import (
    ALL_VECTOR_OPS,
    VECTOR_OP_CATEGORIES,
    VECTOR_OP_CATEGORY_GROUPS,
    AtomicKind,
    AtomicOrdering,
    AtomicScope,
    FloatAssumptionFlags,
    FloatDot4F8Kind,
    IntegerDot4Kind,
    IntegerDot8I4Kind,
    QuantizeNaN,
    QuantizeTie,
    VectorFragmentRole,
    vector_ops,
)
from loom.dsl import ANY, ENCODING_SCHEMA, ENCODING_TRANSFORM, FLOAT, I1, INDEX, INTEGER, SCALAR, VECTOR, EffectKind, Op

_REPO_ROOT = Path(__file__).resolve().parents[5]


SCALAR_TO_VECTOR_SUFFIX_EXCLUSIONS = {
    "assume": "scalar.assume refines scalar predicate facts, not vector lanes.",
}


VECTOR_TO_SCALAR_DESCRIPTOR_EXCLUSIONS = {
    **SCALAR_TO_VECTOR_SUFFIX_EXCLUSIONS,
    "constant": "vector.constant is an aggregate producer, not a lane op.",
    "poison": "vector.poison is a boundary sentinel, not a lane op.",
}


VECTOR_TO_SCALAR_REGISTER_AGGREGATE_MARKERS = {
    "vector.constant": "loom_vector_to_scalar_lower_static_constant",
    "vector.poison": "loom_vector_to_scalar_lower_static_poison",
    "vector.empty": "loom_vector_empty_isa",
    "vector.broadcast": "LOOM_OP_VECTOR_BROADCAST",
    "vector.from_elements": "loom_vector_to_scalar_try_from_elements_lane",
    "vector.extract": "LOOM_OP_VECTOR_EXTRACT",
    "vector.insert": "LOOM_OP_VECTOR_INSERT",
    "vector.slice": "LOOM_OP_VECTOR_SLICE",
    "vector.concat": "LOOM_OP_VECTOR_CONCAT",
    "vector.transpose": "LOOM_OP_VECTOR_TRANSPOSE",
    "vector.shuffle": "LOOM_OP_VECTOR_SHUFFLE",
    "vector.interleave": "LOOM_OP_VECTOR_INTERLEAVE",
    "vector.deinterleave": "LOOM_OP_VECTOR_DEINTERLEAVE",
    "vector.bitcast": "LOOM_OP_VECTOR_BITCAST",
}

VECTOR_TO_SCALAR_SOURCE_FILES = (
    "loom/src/loom/passes/vector_to_scalar.c",
    "loom/src/loom/passes/vector_to_scalar_aggregates.c",
    "loom/src/loom/passes/vector_to_scalar_descriptors.c",
    "loom/src/loom/passes/vector_to_scalar_lanes.c",
    "loom/src/loom/passes/vector_to_scalar_quantized.c",
    "loom/src/loom/passes/vector_to_scalar_reductions.c",
    "loom/src/loom/passes/vector_to_scalar_structural.c",
    "loom/src/loom/passes/vector_to_scalar_tables.c",
    "loom/src/loom/passes/vector_to_scalar_terms.c",
)


def _op_by_name() -> dict[str, Op]:
    return {op.name: op for op in ALL_VECTOR_OPS}


def test_vector_op_category_groups_cover_registry_once() -> None:
    grouped_ops = [op for _, category_ops in VECTOR_OP_CATEGORY_GROUPS for op in category_ops]
    grouped_names = [op.name for op in grouped_ops]

    assert tuple(grouped_ops) == ALL_VECTOR_OPS
    assert len(grouped_names) == len(set(grouped_names))
    assert tuple(category for category, _ in VECTOR_OP_CATEGORY_GROUPS) == VECTOR_OP_CATEGORIES
    assert vector_ops.categories == VECTOR_OP_CATEGORIES


def _assert_optional_cache_policy_attrs(op: Op) -> None:
    cache_scope = op.attr("cache_scope")
    cache_temporal = op.attr("cache_temporal")
    assert cache_scope is not None
    assert cache_temporal is not None
    assert cache_scope.optional
    assert cache_temporal.optional
    assert cache_scope.enum_def is CacheScope
    assert cache_temporal.enum_def is CacheTemporal
    assert CacheScope.c_type == "loom_cache_scope_t"
    assert CacheTemporal.c_type == "loom_cache_temporal_t"


def _op_suffix(op: Op) -> str:
    return op.name.split(".", 1)[1]


def _enum_name(op_name: str) -> str:
    return "LOOM_OP_" + op_name.upper().replace(".", "_")


def _read_repo_file(path: str) -> str:
    return (_REPO_ROOT / path).read_text()


def _vector_to_scalar_all_sources() -> str:
    return "\n".join(_read_repo_file(path) for path in VECTOR_TO_SCALAR_SOURCE_FILES)


def _vector_to_scalar_descriptor_table_source() -> str:
    source = _read_repo_file("loom/src/loom/passes/vector_to_scalar_descriptors.c")
    table_start = source.index("kVectorToScalarDescriptors[]")
    table_end = source.index("\n};", table_start)
    return source[table_start:table_end]


def test_vector_seed_mirrors_scalar_spelling_for_lanewise_ops() -> None:
    scalar_names = {_op_suffix(op) for op in ALL_SCALAR_OPS}
    vector_names = {_op_suffix(op) for op in ALL_VECTOR_OPS}

    missing = sorted(scalar_names - vector_names - set(SCALAR_TO_VECTOR_SUFFIX_EXCLUSIONS))
    unknown_exclusions = sorted(set(SCALAR_TO_VECTOR_SUFFIX_EXCLUSIONS) - scalar_names)
    stale_exclusions = sorted(set(SCALAR_TO_VECTOR_SUFFIX_EXCLUSIONS) & vector_names)

    assert not missing, f"scalar ops missing vector mirrors: {missing}"
    assert not unknown_exclusions, f"unknown scalar/vector exclusions: {unknown_exclusions}"
    assert not stale_exclusions, f"stale scalar/vector exclusions: {stale_exclusions}"
    assert "sqrt" not in vector_names
    assert "fpext" not in vector_names


def test_scalar_vector_mirrors_have_authoring_docs_and_formats() -> None:
    scalar_names = {_op_suffix(op) for op in ALL_SCALAR_OPS}
    vector_ops = _op_by_name()
    vector_names = {_op_suffix(op) for op in ALL_VECTOR_OPS}

    mirror_names = sorted((scalar_names & vector_names) - set(SCALAR_TO_VECTOR_SUFFIX_EXCLUSIONS))
    missing_docs = [f"vector.{name}" for name in mirror_names if not vector_ops[f"vector.{name}"].doc.strip()]
    missing_formats = [f"vector.{name}" for name in mirror_names if not vector_ops[f"vector.{name}"].format]

    assert not missing_docs, f"vector mirrors missing docs: {missing_docs}"
    assert not missing_formats, f"vector mirrors missing formats: {missing_formats}"


def test_vector_registry_matches_generated_c_header() -> None:
    header = _read_repo_file("loom/src/loom/ops/vector/ops.h")
    missing_enums = [op.name for op in ALL_VECTOR_OPS if _enum_name(op.name) not in header]

    assert not missing_enums, f"vector ops missing generated C enums: {missing_enums}"


def test_vector_roundtrip_file_covers_every_vector_op() -> None:
    source = _read_repo_file("loom/src/loom/ops/vector/test/roundtrip.loom-test")
    body = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("//"))
    missing_ops = [op.name for op in ALL_VECTOR_OPS if op.name not in body]

    assert not missing_ops, f"vector ops missing text round-trip coverage: {missing_ops}"


def test_scalar_vector_lanewise_mirrors_have_scalarization_descriptors() -> None:
    scalar_names = {_op_suffix(op) for op in ALL_SCALAR_OPS}
    vector_names = {_op_suffix(op) for op in ALL_VECTOR_OPS}
    descriptor_source = _vector_to_scalar_descriptor_table_source()

    expected = sorted((scalar_names & vector_names) - set(VECTOR_TO_SCALAR_DESCRIPTOR_EXCLUSIONS))
    missing_descriptors = [f"vector.{name}" for name in expected if _enum_name(f"vector.{name}") not in descriptor_source]
    unknown_exclusions = sorted(set(VECTOR_TO_SCALAR_DESCRIPTOR_EXCLUSIONS) - scalar_names)
    stale_exclusions = [f"vector.{name}" for name in sorted(set(VECTOR_TO_SCALAR_DESCRIPTOR_EXCLUSIONS) & vector_names) if _enum_name(f"vector.{name}") in descriptor_source]

    assert not missing_descriptors, f"vector mirrors missing vector-to-scalar descriptors: {missing_descriptors}"
    assert not unknown_exclusions, f"unknown vector-to-scalar descriptor exclusions: {unknown_exclusions}"
    assert not stale_exclusions, f"vector-to-scalar descriptor exclusions now have descriptors and should be removed: {stale_exclusions}"


def test_register_aggregate_ops_have_vector_to_scalar_coverage() -> None:
    source = _vector_to_scalar_all_sources()
    vector_names = {op.name for op in ALL_VECTOR_OPS}

    unknown = sorted(set(VECTOR_TO_SCALAR_REGISTER_AGGREGATE_MARKERS) - vector_names)
    missing = [op_name for op_name, marker in sorted(VECTOR_TO_SCALAR_REGISTER_AGGREGATE_MARKERS.items()) if marker not in source]

    assert not unknown, f"unknown vector aggregate scalarization coverage entries: {unknown}"
    assert not missing, f"vector aggregate ops missing vector-to-scalar coverage: {missing}"


def test_poison_ops_are_pure_typed_sentinels_not_constants() -> None:
    scalar_op = {op.name: op for op in ALL_SCALAR_OPS}["scalar.poison"]
    vector_op = _op_by_name()["vector.poison"]

    assert scalar_op.operands == ()
    assert vector_op.operands == ()
    assert scalar_op.results[0].type_constraint == SCALAR
    assert vector_op.results[0].type_constraint == VECTOR
    assert "Pure" in {trait.name for trait in scalar_op.traits}
    assert "Pure" in {trait.name for trait in vector_op.traits}
    assert "ConstantLike" not in {trait.name for trait in scalar_op.traits}
    assert "ConstantLike" not in {trait.name for trait in vector_op.traits}
    assert "zero-lane vector" in vector_op.doc


def test_empty_vector_is_constant_like_empty_aggregate() -> None:
    op = _op_by_name()["vector.empty"]

    assert op.operands == ()
    assert op.results[0].type_constraint == VECTOR
    trait_names = {trait.name for trait in op.traits}
    assert "Pure" in trait_names
    assert "ConstantLike" in trait_names
    assert "not poison" in op.doc


def test_vector_coordinate_construction_is_pure_and_explicit() -> None:
    ops = _op_by_name()
    iota = ops["vector.iota"]
    mask_range = ops["vector.mask.range"]
    iota_constraints = {(constraint.name, constraint.args) for constraint in iota.constraints}
    mask_range_constraints = {(constraint.name, constraint.args) for constraint in mask_range.constraints}

    assert ("SameType", ("base", "step")) in iota_constraints
    assert ("SameElementType", ("base", "step", "result")) in iota_constraints
    assert "base + i * step" in iota.doc
    assert "Dynamic result extents are allowed" in iota.doc

    assert ("HasI1Element", ("result",)) in mask_range_constraints
    assert ("SameType", ("lower_bound", "upper_bound", "step")) in mask_range_constraints
    assert "inclusive-lower, exclusive-upper" in mask_range.doc
    assert "scf.for" in mask_range.doc

    for op in (iota, mask_range):
        assert "Pure" in {trait.name for trait in op.traits}
        assert "Elementwise" not in {trait.name for trait in op.traits}
        assert op.effects == ()


def test_vector_ctpopi_is_lanewise_integer_popcount() -> None:
    op = _op_by_name()["vector.ctpopi"]
    constraints = {(constraint.name, constraint.args) for constraint in op.constraints}
    trait_names = {trait.name for trait in op.traits}

    assert ("HasIntegerElement", ("result",)) in constraints
    assert ("SameType", ("input", "result")) in constraints
    assert "Pure" in trait_names
    assert "Elementwise" in trait_names
    assert op.effects == ()


def test_vector_fields_do_not_use_scalar_family_constraints() -> None:
    forbidden = {FLOAT, INTEGER, I1}
    for op in ALL_VECTOR_OPS:
        for field in (*op.operands, *op.results):
            assert field.type_constraint not in forbidden, (op.name, field.name)


def test_mask_ops_keep_vector_shape_explicit() -> None:
    ops = _op_by_name()

    cmpi_constraints = {(constraint.name, constraint.args) for constraint in ops["vector.cmpi"].constraints}
    cmpf_constraints = {(constraint.name, constraint.args) for constraint in ops["vector.cmpf"].constraints}
    select_constraints = {(constraint.name, constraint.args) for constraint in ops["vector.select"].constraints}

    assert ("HasI1Element", ("result",)) in cmpi_constraints
    assert ("HasI1Element", ("result",)) in cmpf_constraints
    assert ("HasI1Element", ("condition",)) in select_constraints
    assert ("SameShape", ("lhs", "rhs", "result")) in cmpi_constraints
    assert ("SameShape", ("lhs", "rhs", "result")) in cmpf_constraints


def test_vector_float_flags_are_assumptions_not_fastmath() -> None:
    assert [case.keyword for case in FloatAssumptionFlags.cases] == [
        "nnan",
        "ninf",
        "nsz",
    ]


def test_vector_wave1_float_math_ops_are_lanewise_and_assumption_flagged() -> None:
    ops = _op_by_name()
    for name in (
        "vector.addf",
        "vector.subf",
        "vector.mulf",
        "vector.divf",
        "vector.remf",
        "vector.minimumf",
        "vector.maximumf",
        "vector.minnumf",
        "vector.maxnumf",
        "vector.copysignf",
        "vector.powf",
        "vector.atan2f",
    ):
        op = ops[name]
        constraints = {(constraint.name, constraint.args) for constraint in op.constraints}
        assert ("HasFloatElement", ("result",)) in constraints
        assert ("SameType", ("lhs", "rhs", "result")) in constraints
        assert "Pure" in {trait.name for trait in op.traits}
        assert "Elementwise" in {trait.name for trait in op.traits}
        assert op.attrs[0].name == "assumptions"

    for name in (
        "vector.negf",
        "vector.absf",
        "vector.expf",
        "vector.exp2f",
        "vector.expm1f",
        "vector.logf",
        "vector.log2f",
        "vector.log10f",
        "vector.log1pf",
        "vector.sqrtf",
        "vector.rsqrtf",
        "vector.cbrtf",
        "vector.sinf",
        "vector.cosf",
        "vector.tanf",
        "vector.asinf",
        "vector.acosf",
        "vector.atanf",
        "vector.sinhf",
        "vector.coshf",
        "vector.tanhf",
        "vector.asinhf",
        "vector.acoshf",
        "vector.atanhf",
        "vector.erff",
        "vector.erfcf",
        "vector.ceilf",
        "vector.floorf",
        "vector.roundf",
        "vector.roundevenf",
        "vector.truncf",
    ):
        op = ops[name]
        constraints = {(constraint.name, constraint.args) for constraint in op.constraints}
        assert ("HasFloatElement", ("result",)) in constraints
        assert ("SameType", ("input", "result")) in constraints
        assert "Pure" in {trait.name for trait in op.traits}
        assert "Elementwise" in {trait.name for trait in op.traits}
        assert op.attrs[0].name == "assumptions"


def test_vector_wave2_integer_ops_are_lanewise_and_flagged() -> None:
    ops = _op_by_name()
    for name in (
        "vector.addi",
        "vector.subi",
        "vector.muli",
        "vector.shli",
    ):
        op = ops[name]
        constraints = {(constraint.name, constraint.args) for constraint in op.constraints}
        assert ("HasIntegerElement", ("result",)) in constraints
        assert ("SameType", ("lhs", "rhs", "result")) in constraints
        assert "Pure" in {trait.name for trait in op.traits}
        assert "Elementwise" in {trait.name for trait in op.traits}
        assert op.attrs[0].name == "overflow"

    for name in (
        "vector.divsi",
        "vector.divui",
        "vector.remsi",
        "vector.remui",
        "vector.ceildivsi",
        "vector.ceildivui",
        "vector.floordivsi",
        "vector.minsi",
        "vector.maxsi",
        "vector.minui",
        "vector.maxui",
        "vector.andi",
        "vector.ori",
        "vector.xori",
        "vector.shrsi",
        "vector.shrui",
        "vector.rotli",
        "vector.rotri",
    ):
        op = ops[name]
        constraints = {(constraint.name, constraint.args) for constraint in op.constraints}
        assert ("HasIntegerElement", ("result",)) in constraints
        assert ("SameType", ("lhs", "rhs", "result")) in constraints
        assert "Pure" in {trait.name for trait in op.traits}
        assert "Elementwise" in {trait.name for trait in op.traits}

    for name in (
        "vector.negi",
        "vector.absi",
        "vector.ctlzi",
        "vector.cttzi",
        "vector.ctpopi",
        "vector.signi",
    ):
        op = ops[name]
        constraints = {(constraint.name, constraint.args) for constraint in op.constraints}
        assert ("HasIntegerElement", ("result",)) in constraints
        assert ("SameType", ("input", "result")) in constraints
        assert "Pure" in {trait.name for trait in op.traits}
        assert "Elementwise" in {trait.name for trait in op.traits}

    fmai = ops["vector.fmai"]
    constraints = {(constraint.name, constraint.args) for constraint in fmai.constraints}
    assert ("HasIntegerElement", ("result",)) in constraints
    assert ("SameType", ("a", "b", "c", "result")) in constraints
    assert fmai.attrs[0].name == "overflow"


def test_vector_wave2_predicate_ops_preserve_shape_and_return_masks() -> None:
    ops = _op_by_name()
    for name in ("vector.isnanf", "vector.isinff", "vector.isfinitef"):
        op = ops[name]
        constraints = {(constraint.name, constraint.args) for constraint in op.constraints}
        assert ("HasFloatElement", ("input",)) in constraints
        assert ("HasI1Element", ("result",)) in constraints
        assert ("SameKind", ("input", "result")) in constraints
        assert ("SameShape", ("input", "result")) in constraints
        assert "Pure" in {trait.name for trait in op.traits}
        assert "Elementwise" in {trait.name for trait in op.traits}

    signf = ops["vector.signf"]
    constraints = {(constraint.name, constraint.args) for constraint in signf.constraints}
    assert ("HasFloatElement", ("result",)) in constraints
    assert ("SameType", ("input", "result")) in constraints


def test_vector_atomic_attrs_are_shared_c_enum_aliases() -> None:
    assert AtomicOrdering.c_type == "loom_atomic_ordering_t"
    assert AtomicScope.c_type == "loom_atomic_scope_t"
    assert AtomicKind.c_type == "loom_atomic_kind_t"
    assert [case.keyword for case in AtomicOrdering.cases] == [
        "relaxed",
        "acquire",
        "release",
        "acq_rel",
        "seq_cst",
    ]
    assert [case.keyword for case in AtomicScope.cases] == [
        "thread",
        "subgroup",
        "workgroup",
        "device",
        "system",
    ]
    assert [case.keyword for case in AtomicKind.cases] == [
        "xchgi",
        "xchgf",
        "addi",
        "addf",
        "subi",
        "andi",
        "ori",
        "xori",
        "minsi",
        "maxsi",
        "minui",
        "maxui",
        "minimumf",
        "maximumf",
        "minnumf",
        "maxnumf",
    ]


def test_vector_memory_ops_are_effectful_and_view_based() -> None:
    ops = _op_by_name()

    assert ops["vector.load"].effects[0].operand == "view"
    assert ops["vector.load"].effects[0].kind == EffectKind.READ
    assert ops["vector.load.mask"].effects[0].operand == "view"
    assert ops["vector.load.mask"].effects[0].kind == EffectKind.READ
    assert ops["vector.store"].effects[0].operand == "view"
    assert ops["vector.store"].effects[0].kind == EffectKind.WRITE
    assert ops["vector.store.mask"].effects[0].operand == "view"
    assert ops["vector.store.mask"].effects[0].kind == EffectKind.WRITE
    assert ops["vector.load.expand"].effects[0].operand == "view"
    assert ops["vector.load.expand"].effects[0].kind == EffectKind.READ
    assert ops["vector.store.compress"].effects[0].operand == "view"
    assert ops["vector.store.compress"].effects[0].kind == EffectKind.WRITE
    load_expand_constraints = {(constraint.name, constraint.args) for constraint in ops["vector.load.expand"].constraints}
    store_compress_constraints = {(constraint.name, constraint.args) for constraint in ops["vector.store.compress"].constraints}
    assert ("HasRankOneVector", ("result",)) in load_expand_constraints
    assert ("HasRankOneVector", ("value",)) in store_compress_constraints
    offset_indexed_ops = (
        "vector.gather",
        "vector.gather.mask",
        "vector.scatter",
        "vector.scatter.mask",
        "vector.atomic.reduce",
        "vector.atomic.reduce.mask",
        "vector.atomic.rmw",
        "vector.atomic.rmw.mask",
        "vector.atomic.cmpxchg",
    )
    for name in offset_indexed_ops:
        constraints = {(constraint.name, constraint.args) for constraint in ops[name].constraints}
        assert ("HasIndexOrNonI1IntegerElement", ("offsets",)) in constraints
    assert ops["vector.gather"].effects[0].operand == "view"
    assert ops["vector.gather"].effects[0].kind == EffectKind.READ
    assert ops["vector.gather.mask"].effects[0].operand == "view"
    assert ops["vector.gather.mask"].effects[0].kind == EffectKind.READ
    assert ops["vector.scatter"].effects[0].operand == "view"
    assert ops["vector.scatter"].effects[0].kind == EffectKind.WRITE
    assert ops["vector.scatter.mask"].effects[0].operand == "view"
    assert ops["vector.scatter.mask"].effects[0].kind == EffectKind.WRITE
    assert ops["vector.atomic.reduce"].effects[0].operand == "view"
    assert ops["vector.atomic.reduce"].effects[0].kind == EffectKind.READWRITE
    assert ops["vector.atomic.reduce.mask"].effects[0].operand == "view"
    assert ops["vector.atomic.reduce.mask"].effects[0].kind == EffectKind.READWRITE
    assert ops["vector.atomic.rmw"].effects[0].operand == "view"
    assert ops["vector.atomic.rmw"].effects[0].kind == EffectKind.READWRITE
    assert ops["vector.atomic.rmw.mask"].effects[0].operand == "view"
    assert ops["vector.atomic.rmw.mask"].effects[0].kind == EffectKind.READWRITE
    assert ops["vector.atomic.cmpxchg"].effects[0].operand == "view"
    assert ops["vector.atomic.cmpxchg"].effects[0].kind == EffectKind.READWRITE
    cmpxchg_constraints = {(constraint.name, constraint.args) for constraint in ops["vector.atomic.cmpxchg"].constraints}
    assert ("HasIndexOrNonI1IntegerElement", ("expected",)) in cmpxchg_constraints
    assert ("SameShape", ("offsets", "expected", "replacement", "old")) in cmpxchg_constraints

    for name in (
        "vector.load",
        "vector.load.mask",
        "vector.store",
        "vector.store.mask",
        "vector.load.expand",
        "vector.store.compress",
        "vector.gather",
        "vector.gather.mask",
        "vector.scatter",
        "vector.scatter.mask",
        "vector.atomic.reduce",
        "vector.atomic.reduce.mask",
        "vector.atomic.rmw",
        "vector.atomic.rmw.mask",
        "vector.atomic.cmpxchg",
    ):
        assert "Pure" not in {trait.name for trait in ops[name].traits}
        _assert_optional_cache_policy_attrs(ops[name])


def test_vector_static_construction_ops_declare_shape_constraints() -> None:
    ops = _op_by_name()
    from_elements_constraints = {(constraint.name, constraint.args) for constraint in ops["vector.from_elements"].constraints}
    shuffle_constraints = {(constraint.name, constraint.args) for constraint in ops["vector.shuffle"].constraints}

    assert ("HasAllStaticVector", ("result",)) in from_elements_constraints
    assert ("SameElementType", ("elements", "result")) in from_elements_constraints
    assert (
        "ValueCountMatchesStaticElementCount",
        ("result", "elements"),
    ) in from_elements_constraints
    assert ("HasAllStaticRankOneVector", ("source",)) in shuffle_constraints
    assert ("SameType", ("source", "result")) in shuffle_constraints


def test_vector_coordinate_ops_declare_index_or_integer_constraints() -> None:
    ops = _op_by_name()
    iota_constraints = {(constraint.name, constraint.args) for constraint in ops["vector.iota"].constraints}
    mask_range_constraints = {(constraint.name, constraint.args) for constraint in ops["vector.mask.range"].constraints}

    assert ("HasIndexOrNonI1IntegerScalar", ("base",)) in iota_constraints
    assert ("HasIndexOrNonI1IntegerScalar", ("lower_bound",)) in mask_range_constraints


def test_vector_table_lookup_is_pure_register_lookup() -> None:
    op = _op_by_name()["vector.table.lookup"]
    constraints = {(constraint.name, constraint.args) for constraint in op.constraints}

    assert ("HasRankOneVector", ("table",)) in constraints
    assert ("HasIndexOrNonI1IntegerElement", ("indices",)) in constraints
    assert ("SameElementType", ("table", "result")) in constraints
    assert ("SameShape", ("indices", "result")) in constraints
    assert "Pure" in {trait.name for trait in op.traits}
    assert op.effects == ()


def test_vector_table_quantize_is_explicit_register_encode() -> None:
    op = _op_by_name()["vector.table.quantize"]
    constraints = {(constraint.name, constraint.args) for constraint in op.constraints}

    assert [case.keyword for case in QuantizeNaN.cases] == ["zero", "max"]
    assert [case.keyword for case in QuantizeTie.cases] == ["lower", "upper"]
    assert ("HasFloatElement", ("input",)) in constraints
    assert ("HasFloatElement", ("thresholds",)) in constraints
    assert ("HasRankOneVector", ("thresholds",)) in constraints
    assert ("HasIntegerElement", ("result",)) in constraints
    assert ("SameElementType", ("input", "thresholds")) in constraints
    assert ("SameShape", ("input", "result")) in constraints
    assert "Pure" in {trait.name for trait in op.traits}
    assert op.effects == ()


def test_vector_transform_is_pure_numeric_transform_boundary() -> None:
    op = _op_by_name()["vector.transform"]
    constraints = {(constraint.name, constraint.args) for constraint in op.constraints}
    trait_names = {trait.name for trait in op.traits}
    transform_operand = op.operand("transform")

    assert transform_operand is not None
    assert transform_operand.type_constraint == ENCODING_TRANSFORM
    assert ("HasFloatElement", ("source",)) in constraints
    assert ("HasFloatElement", ("result",)) in constraints
    assert ("SameElementType", ("source", "result")) in constraints
    assert "Pure" in trait_names
    assert "Elementwise" not in trait_names
    assert op.effects == ()


def test_vector_encode_decode_keep_schema_and_data_separate() -> None:
    ops = _op_by_name()

    for name, data_operand_name in (
        ("vector.decode", "payload"),
        ("vector.encode", "source"),
    ):
        op = ops[name]
        trait_names = {trait.name for trait in op.traits}
        data_operand = op.operand(data_operand_name)
        schema_operand = op.operand("schema")
        auxiliary_operand = op.operand("auxiliary")
        auxiliary_names_attr = op.attr("auxiliary_names")

        assert data_operand is not None
        assert data_operand.type_constraint == VECTOR
        assert schema_operand is not None
        assert schema_operand.type_constraint == ENCODING_SCHEMA
        assert auxiliary_operand is not None
        assert auxiliary_operand.type_constraint == VECTOR
        assert auxiliary_operand.variadic
        assert auxiliary_names_attr is not None
        assert auxiliary_names_attr.attr_type == "dict"
        assert auxiliary_names_attr.optional
        assert any(isinstance(element, OperandDict) and element.operands == "auxiliary" and element.names == "auxiliary_names" for element in op.format)
        assert "Pure" in trait_names
        assert "Elementwise" not in trait_names
        assert op.effects == ()


def test_vector_fragment_preserves_physical_type_and_groups_interpretation_data() -> None:
    op = _op_by_name()["vector.fragment"]
    trait_names = {trait.name for trait in op.traits}
    constraints = {(constraint.name, constraint.args) for constraint in op.constraints}

    assert [case.keyword for case in VectorFragmentRole.cases] == ["lhs", "rhs", "init", "result"]
    role_attr = op.attr("role")
    data_operand = op.operand("data")
    rows_operand = op.operand("rows")
    columns_operand = op.operand("columns")
    params_operand = op.operand("params")
    result = op.result("result")
    assert role_attr is not None
    assert data_operand is not None
    assert rows_operand is not None
    assert columns_operand is not None
    assert params_operand is not None
    assert result is not None
    assert role_attr.enum_def is VectorFragmentRole
    assert data_operand.type_constraint == VECTOR
    assert rows_operand.type_constraint == INDEX
    assert columns_operand.type_constraint == INDEX
    assert params_operand.type_constraint == ANY
    assert params_operand.variadic
    assert result.type_constraint == VECTOR
    assert ("SameType", ("data", "result")) in constraints
    assert "Pure" in trait_names
    assert "Elementwise" not in trait_names
    assert op.effects == ()

    param_names_attr = op.attr("param_names")
    predicates_attr = op.attr("predicates")
    assert param_names_attr is not None
    assert param_names_attr.attr_type == "dict"
    assert param_names_attr.optional
    assert predicates_attr is not None
    assert predicates_attr.attr_type == "predicate_list"
    assert predicates_attr.optional

    using_groups = [
        element
        for element in op.format
        if isinstance(element, OptionalGroup) and any(isinstance(inner, OperandDict) and inner.operands == "params" and inner.names == "param_names" for inner in element.elements)
    ]
    where_groups = [element for element in op.format if isinstance(element, OptionalGroup) and any(isinstance(inner, PredicateList) and inner.field == "predicates" for inner in element.elements)]
    assert len(using_groups) == 1
    assert len(where_groups) == 1
    assert any(isinstance(element, ResultType) and element.field == "result" for element in op.format)


def test_vector_layout_ops_make_lane_structure_explicit() -> None:
    ops = _op_by_name()
    interleave_constraints = {(constraint.name, constraint.args) for constraint in ops["vector.interleave"].constraints}
    deinterleave_constraints = {(constraint.name, constraint.args) for constraint in ops["vector.deinterleave"].constraints}

    assert ("SameType", ("even", "odd")) in interleave_constraints
    assert ("SameElementType", ("even", "result")) in interleave_constraints
    assert ("SameType", ("results",)) in deinterleave_constraints
    assert ("SameElementType", ("source", "results")) in deinterleave_constraints

    for name in ("vector.interleave", "vector.deinterleave"):
        trait_names = {trait.name for trait in ops[name].traits}
        assert "Pure" in trait_names
        assert "Elementwise" not in trait_names
        assert ops[name].effects == ()


def test_vector_bitcast_is_whole_register_not_lanewise_only() -> None:
    op = _op_by_name()["vector.bitcast"]
    constraints = {(constraint.name, constraint.args) for constraint in op.constraints}
    trait_names = {trait.name for trait in op.traits}

    assert ("SameShape", ("input", "result")) not in constraints
    assert ("TotalBitCountEqual", ("input", "result")) in constraints
    assert "Pure" in trait_names
    assert "Elementwise" not in trait_names


def test_vector_constant_payload_type_is_generated_constraint() -> None:
    op = _op_by_name()["vector.constant"]
    constraints = {(constraint.name, constraint.args) for constraint in op.constraints}

    assert ("AttrMatchesElementType", ("value", "result")) in constraints
    assert not op.verify


def test_vector_bitfield_ops_are_pure_integer_register_ops() -> None:
    ops = _op_by_name()

    extractu_constraints = {(constraint.name, constraint.args) for constraint in ops["vector.bitfield.extractu"].constraints}
    extracts_constraints = {(constraint.name, constraint.args) for constraint in ops["vector.bitfield.extracts"].constraints}
    insert_constraints = {(constraint.name, constraint.args) for constraint in ops["vector.bitfield.insert"].constraints}

    assert ("HasIntegerElement", ("source",)) in extractu_constraints
    assert ("HasIntegerElement", ("result",)) in extractu_constraints
    assert ("SameShape", ("source", "result")) in extractu_constraints
    assert ("HasIntegerElement", ("source",)) in extracts_constraints
    assert ("HasIntegerElement", ("result",)) in extracts_constraints
    assert ("SameShape", ("source", "result")) in extracts_constraints

    assert ("HasIntegerElement", ("field",)) in insert_constraints
    assert ("HasIntegerElement", ("base",)) in insert_constraints
    assert ("HasIntegerElement", ("result",)) in insert_constraints
    assert ("SameShape", ("field", "base", "result")) in insert_constraints
    assert ("SameType", ("base", "result")) in insert_constraints

    for name in (
        "vector.bitfield.extractu",
        "vector.bitfield.extracts",
        "vector.bitfield.insert",
    ):
        trait_names = {trait.name for trait in ops[name].traits}
        assert "Pure" in trait_names
        assert "Elementwise" in trait_names


def test_vector_bitpack_ops_are_pure_register_pack_ops() -> None:
    ops = _op_by_name()

    pack_constraints = {(constraint.name, constraint.args) for constraint in ops["vector.bitpack"].constraints}
    unpacku_constraints = {(constraint.name, constraint.args) for constraint in ops["vector.bitunpacku"].constraints}
    unpacks_constraints = {(constraint.name, constraint.args) for constraint in ops["vector.bitunpacks"].constraints}

    assert ("HasIntegerElement", ("source",)) in pack_constraints
    assert ("HasIntegerElement", ("result",)) in pack_constraints
    assert (
        "PackedPayloadBitCountMatchesStorage",
        ("source", "width", "result", "result"),
    ) in pack_constraints
    assert ("HasIntegerElement", ("source",)) in unpacku_constraints
    assert ("HasIntegerElement", ("result",)) in unpacku_constraints
    assert (
        "UnpackedPayloadBitCountMatchesStorage",
        ("result", "width", "source", "result"),
    ) in unpacku_constraints
    assert ("HasIntegerElement", ("source",)) in unpacks_constraints
    assert ("HasIntegerElement", ("result",)) in unpacks_constraints
    assert (
        "UnpackedPayloadBitCountMatchesStorage",
        ("result", "width", "source", "result"),
    ) in unpacks_constraints

    for name in (
        "vector.bitpack",
        "vector.bitunpacku",
        "vector.bitunpacks",
    ):
        trait_names = {trait.name for trait in ops[name].traits}
        assert "Pure" in trait_names
        assert "Elementwise" not in trait_names
        assert ops[name].effects == ()


def test_vector_dot4i_is_pure_grouped_i8_to_i32_dot() -> None:
    op = _op_by_name()["vector.dot4i"]
    constraints = {(constraint.name, constraint.args) for constraint in op.constraints}
    grouped_constraints = {(constraint.name, constraint.args, constraint.data) for constraint in op.constraints}
    trait_names = {trait.name for trait in op.traits}

    assert [case.keyword for case in IntegerDot4Kind.cases] == [
        "s8s8",
        "u8s8",
        "s8u8",
        "u8u8",
    ]
    assert ("HasI8Element", ("lhs",)) in constraints
    assert ("HasI8Element", ("rhs",)) in constraints
    assert ("HasI32Element", ("acc",)) in constraints
    assert ("SameShape", ("lhs", "rhs")) in constraints
    assert ("SameType", ("acc", "result")) in constraints
    assert ("LastAxisGroupedBy", ("lhs", "result"), 4) in grouped_constraints
    assert "Pure" in trait_names
    assert "Elementwise" not in trait_names
    assert op.effects == ()


def test_vector_dot8i4_is_pure_packed_i4_to_i32_dot() -> None:
    op = _op_by_name()["vector.dot8i4"]
    constraints = {(constraint.name, constraint.args) for constraint in op.constraints}
    trait_names = {trait.name for trait in op.traits}

    assert [case.keyword for case in IntegerDot8I4Kind.cases] == [
        "s4s4",
        "u4s4",
        "s4u4",
        "u4u4",
    ]
    assert ("HasI32Element", ("lhs",)) in constraints
    assert ("SameType", ("lhs", "rhs", "acc", "result")) in constraints
    assert "little-endian pack" in op.doc
    assert "sdot8/udot8/sudot8" in op.doc
    assert "Pure" in trait_names
    assert "Elementwise" not in trait_names
    assert op.effects == ()


def test_vector_dot4f8_is_pure_packed_f8_to_f32_dot() -> None:
    op = _op_by_name()["vector.dot4f8"]
    constraints = {(constraint.name, constraint.args) for constraint in op.constraints}
    trait_names = {trait.name for trait in op.traits}

    assert [case.keyword for case in FloatDot4F8Kind.cases] == [
        "fp8bf8",
        "bf8fp8",
        "fp8fp8",
        "bf8bf8",
    ]
    assert ("HasI32Element", ("lhs",)) in constraints
    assert ("HasF32Element", ("acc",)) in constraints
    assert ("SameType", ("lhs", "rhs")) in constraints
    assert ("SameType", ("acc", "result")) in constraints
    assert ("SameShape", ("lhs", "acc")) in constraints
    assert "little-endian pack" in op.doc
    assert "dot4.f32.fp8/bf8" in op.doc
    assert "Pure" in trait_names
    assert "Elementwise" not in trait_names
    assert op.effects == ()


def test_vector_dot2f_is_pure_grouped_f16_bf16_to_f32_dot() -> None:
    op = _op_by_name()["vector.dot2f"]
    constraints = {(constraint.name, constraint.args) for constraint in op.constraints}
    grouped_constraints = {(constraint.name, constraint.args, constraint.data) for constraint in op.constraints}
    trait_names = {trait.name for trait in op.traits}

    assert ("HasF16OrBf16Element", ("lhs",)) in constraints
    assert ("HasF16OrBf16Element", ("rhs",)) in constraints
    assert ("HasF32Element", ("acc",)) in constraints
    assert ("SameShape", ("lhs", "rhs")) in constraints
    assert ("SameElementType", ("lhs", "rhs")) in constraints
    assert ("SameType", ("acc", "result")) in constraints
    assert ("LastAxisGroupedBy", ("lhs", "result"), 2) in grouped_constraints
    assert "fdot2" in op.doc
    assert "scalar.fmaf" in op.doc
    assert "Pure" in trait_names
    assert "Elementwise" not in trait_names
    assert op.effects == ()


def test_vector_dotf_is_pure_same_element_fused_dot() -> None:
    op = _op_by_name()["vector.dotf"]
    constraints = {(constraint.name, constraint.args) for constraint in op.constraints}
    trait_names = {trait.name for trait in op.traits}

    assert ("HasFloatElement", ("lhs",)) in constraints
    assert ("SameShape", ("lhs", "rhs")) in constraints
    assert ("SameType", ("init", "result")) in constraints
    assert ("SameElementType", ("lhs", "rhs", "init")) in constraints
    assert "scalar.fmaf" in op.doc
    assert "Zero-lane inputs return init" in op.doc
    assert "Pure" in trait_names
    assert "Elementwise" not in trait_names
    assert op.effects == ()
