# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.vector import (
    ALL_VECTOR_OPS,
    AtomicKind,
    AtomicOrdering,
    AtomicScope,
    FloatAssumptionFlags,
    IntegerDot4Kind,
    QuantizeNaN,
    QuantizeTie,
)
from loom.dsl import ENCODING_TRANSFORM, FLOAT, I1, INTEGER, SCALAR, VECTOR, EffectKind, Op


def _op_by_name() -> dict[str, Op]:
    return {op.name: op for op in ALL_VECTOR_OPS}


def test_vector_seed_mirrors_scalar_spelling_for_lanewise_ops() -> None:
    scalar_names = {op.name.removeprefix("scalar.") for op in ALL_SCALAR_OPS}
    vector_names = {op.name.removeprefix("vector.") for op in ALL_VECTOR_OPS}

    mirrored_seed = {
        "addi",
        "addf",
        "absf",
        "acosf",
        "acoshf",
        "asinf",
        "asinhf",
        "atan2f",
        "atanf",
        "atanhf",
        "cbrtf",
        "ceilf",
        "copysignf",
        "cosf",
        "coshf",
        "andi",
        "bitcast",
        "cmpf",
        "cmpi",
        "ctpopi",
        "divf",
        "erfcf",
        "erff",
        "exp2f",
        "expf",
        "expm1f",
        "extf",
        "extsi",
        "extui",
        "floorf",
        "fmaf",
        "fptosi",
        "fptoui",
        "fptrunc",
        "log10f",
        "log1pf",
        "log2f",
        "logf",
        "maximumf",
        "maxnumf",
        "minimumf",
        "minnumf",
        "muli",
        "mulf",
        "negf",
        "ori",
        "powf",
        "remf",
        "roundevenf",
        "roundf",
        "rsqrtf",
        "poison",
        "select",
        "sinf",
        "sinhf",
        "sitofp",
        "sqrtf",
        "subf",
        "tanf",
        "tanhf",
        "truncf",
        "trunci",
        "uitofp",
        "xori",
    }
    assert mirrored_seed <= scalar_names
    assert mirrored_seed <= vector_names
    assert "sqrt" not in vector_names
    assert "fpext" not in vector_names


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


def test_vector_atomic_attrs_are_explicit_enums() -> None:
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
    ):
        assert "Pure" not in {trait.name for trait in ops[name].traits}


def test_vector_table_lookup_is_pure_register_lookup() -> None:
    op = _op_by_name()["vector.table.lookup"]
    constraints = {(constraint.name, constraint.args) for constraint in op.constraints}

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
    assert "Pure" in trait_names
    assert "Elementwise" not in trait_names


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
    assert ("HasIntegerElement", ("source",)) in unpacku_constraints
    assert ("HasIntegerElement", ("result",)) in unpacku_constraints
    assert ("HasIntegerElement", ("source",)) in unpacks_constraints
    assert ("HasIntegerElement", ("result",)) in unpacks_constraints

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
    trait_names = {trait.name for trait in op.traits}

    assert [case.keyword for case in IntegerDot4Kind.cases] == [
        "s8s8",
        "u8s8",
        "s8u8",
        "u8u8",
    ]
    assert ("HasIntegerElement", ("lhs",)) in constraints
    assert ("HasIntegerElement", ("rhs",)) in constraints
    assert ("HasIntegerElement", ("acc",)) in constraints
    assert ("SameShape", ("lhs", "rhs")) in constraints
    assert ("SameType", ("acc", "result")) in constraints
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
