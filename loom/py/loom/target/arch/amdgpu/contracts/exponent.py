# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact exponent primitives over native floating instructions."""

from loom.dialect.scalar import math as scalar
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.amdgpu.contracts.materializers import (
    REGISTERS_VGPR_MATERIALIZER,
)
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorOperandMaterialization,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    SourceValueKind,
    TypePattern,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import DescriptorSet

EXPONENT_DESCRIPTOR_KEYS = (
    *(
        f"amdgpu.v_{operation}_f{width}"
        for width in (16, 32, 64)
        for operation in ("ldexp", "frexp_mant", f"frexp_exp_i{min(width, 32)}")
    ),
    "amdgpu.v_min_i32.src0_inline",
    "amdgpu.v_max_i32.lit",
)


def _exponent_rule(
    descriptors: DescriptorSet,
    source_op: Op,
    float_type: TypePattern,
    exponent_type: TypePattern,
    width: int,
    *,
    decompose: bool,
) -> DescriptorRule:
    form = (
        DescriptorEmitForm.PER_LANE
        if float_type.kind == "vector"
        else DescriptorEmitForm.OP
    )
    guards = [
        Guard.value_type("input", float_type),
        Guard.value_type("fraction" if decompose else "result", float_type),
        Guard.value_type("exponent", exponent_type),
    ]
    emit = []

    def instruction(
        name: str,
        result: ValueRef,
        operands: dict[str, ValueRef],
        *,
        immediates: dict[str, int] | None = None,
        materialize_sources: bool = False,
    ) -> None:
        descriptor = descriptor_by_key(descriptors, f"amdgpu.{name}")
        guards.append(Guard.descriptor_available(descriptor))
        emit.append(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands=operands,
                results={"dst": result},
                result_types=(
                    {"dst": DescriptorResultType()}
                    if result.kind == SourceValueKind.TEMPORARY
                    else None
                ),
                immediates=immediates or {},
                form=form,
                operand_materialization=(
                    DescriptorOperandMaterialization.TARGET
                    if materialize_sources
                    else DescriptorOperandMaterialization.DIRECT
                ),
            )
        )

    source = ValueRef.operand("input")
    if width == 16:
        guards.append(
            Guard.value_materializable("input", REGISTERS_VGPR_MATERIALIZER.name)
        )
        source = ValueRef.operand(
            "input", materializer=REGISTERS_VGPR_MATERIALIZER.name
        )
    if decompose:
        instruction(
            f"v_frexp_mant_f{width}", ValueRef.result("fraction"), {"input": source}
        )
        instruction(
            f"v_frexp_exp_i{min(width, 32)}_f{width}",
            ValueRef.temporary("exponent")
            if width == 16
            else ValueRef.result("exponent"),
            {"input": source},
        )
        if width == 16:
            instruction(
                "v_bfe_i32.offset_0_width_inline_low16",
                ValueRef.result("exponent"),
                {"value": ValueRef.temporary("exponent")},
                immediates={"offset": 0, "width": 16},
            )
    else:
        exponent = ValueRef.operand("exponent")
        if width == 16:
            guards.append(
                Guard.value_materializable("exponent", REGISTERS_VGPR_MATERIALIZER.name)
            )
            exponent = ValueRef.operand(
                "exponent", materializer=REGISTERS_VGPR_MATERIALIZER.name
            )
            # Every finite F16 input has underflowed or overflowed beyond
            # these bounds. Clamping before the native i16 interpretation
            # preserves the full public i32 domain, including INT_MIN/MAX.
            instruction(
                "v_max_i32.lit",
                ValueRef.temporary("bounded_low"),
                {"rhs": exponent},
                immediates={"imm32": 0xFFFFFFC0},
            )
            instruction(
                "v_min_i32.src0_inline",
                ValueRef.temporary("bounded"),
                {"rhs": ValueRef.temporary("bounded_low")},
                immediates={"imm32": 64},
            )
            exponent = ValueRef.temporary("bounded")
        instruction(
            f"v_ldexp_f{width}",
            ValueRef.result("result"),
            {"input": source, "exponent": exponent},
            materialize_sources=True,
        )
    return DescriptorRule(
        source_op=source_op,
        descriptor=emit[0].descriptor,
        guards=tuple(guards),
        emit=tuple(emit),
    )


def exponent_rules(descriptors: DescriptorSet) -> tuple[DescriptorRule, ...]:
    rules = []
    for width in (16, 32, 64):
        domains = [(scalar, Scalar(f"f{width}"), Scalar("i32"))]
        if width != 16:
            # Narrow packed vectors use shared exact widening. Each native
            # F32/F64 lane instead occupies a whole register or register pair.
            maximum = (
                "LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES"
                if width == 32
                else "(LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES / 2u)"
            )
            domains.append(
                (
                    vector,
                    Vector(
                        f"f{width}",
                        minimum_static_elements=1,
                        maximum_static_elements=maximum,
                    ),
                    Vector(
                        "i32",
                        minimum_static_elements=1,
                        maximum_static_elements=maximum,
                    ),
                )
            )
        for dialect, float_type, exponent_type in domains:
            prefix = "vector" if dialect is vector else "scalar"
            rules.extend(
                _exponent_rule(
                    descriptors,
                    getattr(dialect, f"{prefix}_{operation}"),
                    float_type,
                    exponent_type,
                    width,
                    decompose=operation == "frexpf",
                )
                for operation in ("ldexpf", "frexpf")
            )
    return tuple(rules)
