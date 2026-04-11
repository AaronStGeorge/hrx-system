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
)
from loom.dsl import FLOAT, I1, INTEGER, EffectKind, Op


def _op_by_name() -> dict[str, Op]:
    return {op.name: op for op in ALL_VECTOR_OPS}


def test_vector_seed_mirrors_scalar_spelling_for_lanewise_ops() -> None:
    scalar_names = {op.name.removeprefix("scalar.") for op in ALL_SCALAR_OPS}
    vector_names = {op.name.removeprefix("vector.") for op in ALL_VECTOR_OPS}

    mirrored_seed = {
        "addi",
        "addf",
        "bitcast",
        "cmpf",
        "cmpi",
        "extf",
        "extsi",
        "extui",
        "fmaf",
        "fptosi",
        "fptoui",
        "fptrunc",
        "muli",
        "mulf",
        "select",
        "sitofp",
        "sqrtf",
        "trunci",
        "uitofp",
    }
    assert mirrored_seed <= scalar_names
    assert mirrored_seed <= vector_names
    assert "sqrt" not in vector_names
    assert "fpext" not in vector_names


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
