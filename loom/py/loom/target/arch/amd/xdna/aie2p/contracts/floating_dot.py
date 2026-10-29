# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P floating-point dot-product selection rules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.contracts.data_path import (
    vector_data_path_control,
)
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    ContractEmit,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterCopy,
    EmitRegisterSlice,
    Guard,
    ResultTypeBinding,
    SourceNode,
    TypePattern,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_BF16_DOT2_VECTOR = Vector(
    "bf16", minimum_static_elements=2, maximum_static_elements=32
)
_BF16X32_VECTOR = Vector("bf16", lanes=32)
_F32_VECTOR = Vector("f32", minimum_static_elements=1, maximum_static_elements=16)
_F32X16_VECTOR = Vector("f32", lanes=16)

_BF16_ELEMENTWISE_MULTIPLY_CONTROL = vector_data_path_control(
    sign_x=False,
    sign_y=False,
    accumulator_mode=2,
    multiplication_mode=3,
    compute_mode=1,
)

# AIE2P T16_32x2_lo/hi select the even and odd BF16 lanes from a 512-bit
# source. Two ordered VMACs over those streams implement vector.dot2f's two
# sequential fused accumulations without weakening its native grouped contract.
_BF16_DOT2_DEINTERLEAVE_CONTROLS = (2, 3)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


def _op_emit(
    descriptor: Descriptor,
    *,
    operands: Mapping[str, ValueRef] | None = None,
    results: Mapping[str, ValueRef] | None = None,
    result_types: Mapping[str, ResultTypeBinding] | None = None,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={} if operands is None else operands,
        results={} if results is None else results,
        result_types=result_types,
        form=DescriptorEmitForm.OP,
    )


def _constant_emit(
    descriptor: Descriptor,
    result: ValueRef,
    value: int,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        results={"dst": result},
        result_types={"dst": DescriptorResultType()},
        immediates={"i": value},
        form=DescriptorEmitForm.CONST,
    )


def _vector_dot2f_bf16_rule(
    input_type: TypePattern,
    result_type: TypePattern,
    *,
    initial_accumulator: Literal["source", "zero"],
    accumulator_in_accumulator: bool,
    result_in_accumulator: bool,
    report_key: str,
    rhs_form: Literal["packed", "interleaved"] = "packed",
) -> DescriptorRule:
    config_constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    shuffle = _descriptor("amd.xdna.aie2p.shuffle.x.configured")
    clear = _descriptor("amd.xdna.aie2p.accumulator.clear.f32x64")
    move_to_accumulator = _descriptor(
        "amd.xdna.aie2p.move.vector512.to.accumulator512.low"
    )
    accumulate = _descriptor("amd.xdna.aie2p.accumulate.bf16x32.configured")
    first_accumulate = (
        accumulate
        if initial_accumulator == "zero"
        else _descriptor("amd.xdna.aie2p.accumulate.bf16x16.configured")
    )
    move_from_accumulator = _descriptor(
        "amd.xdna.aie2p.move.accumulator512.to.vector512"
    )

    emits: list[ContractEmit] = []
    input_values = {
        operand_name: ValueRef.operand(operand_name) for operand_name in ("lhs", "rhs")
    }
    for lane_group, control in zip(
        ("even", "odd"), _BF16_DOT2_DEINTERLEAVE_CONTROLS, strict=True
    ):
        control_value = ValueRef.temporary(f"{lane_group}_control")
        emits.append(_constant_emit(config_constant, control_value, control))
        emits.extend(
            _op_emit(
                shuffle,
                operands={
                    "s1": input_values[operand_name],
                    "s2": input_values[operand_name],
                    "mod": control_value,
                },
                results={"dst": ValueRef.temporary(f"{operand_name}_{lane_group}")},
                result_types={"dst": DescriptorResultType()},
            )
            for operand_name in (
                ("lhs",) if rhs_form == "interleaved" else ("lhs", "rhs")
            )
        )

    # An interleave supplies the low sixteen even/odd lanes directly. Only
    # those accumulator lanes belong to the source result; carrier padding
    # does not participate in the returned dot product.
    rhs_values = {
        lane_group: (
            ValueRef.operand(lane_group, source_node="interleave")
            if rhs_form == "interleaved"
            else ValueRef.temporary(f"rhs_{lane_group}")
        )
        for lane_group in ("even", "odd")
    }
    accumulator = ValueRef.temporary(
        "zero_accumulator" if initial_accumulator == "zero" else "initial_accumulator"
    )
    guards = (
        Guard.value_type("lhs", input_type),
        Guard.value_type("rhs", input_type),
        Guard.value_type("acc", result_type),
        Guard.value_type("result", result_type),
        Guard.low_value_register_class(
            "acc",
            "aie2p.ebmll" if accumulator_in_accumulator else "aie2p.vec256",
        ),
        Guard.low_value_register_class(
            "result",
            "aie2p.ebmll" if result_in_accumulator else "aie2p.vec256",
        ),
    )
    if initial_accumulator == "zero":
        # Exact floating facts distinguish positive zero from negative zero.
        guards += (Guard.value_float_equals("acc", 0.0),)
        emits.append(
            _op_emit(
                clear,
                results={"dst": accumulator},
                result_types={"dst": DescriptorResultType()},
            )
        )
    elif accumulator_in_accumulator:
        accumulator = ValueRef.operand("acc")
    else:
        emits.append(
            _op_emit(
                move_to_accumulator,
                operands={"src": ValueRef.operand("acc")},
                results={"dst": accumulator},
                result_types={"dst": DescriptorResultType()},
            )
        )
    emits.extend(
        (
            _constant_emit(
                config_constant,
                ValueRef.temporary("accumulate_control"),
                _BF16_ELEMENTWISE_MULTIPLY_CONTROL,
            ),
            _op_emit(
                first_accumulate,
                operands={
                    "acc1": accumulator,
                    "s1": ValueRef.temporary("lhs_even"),
                    "s2": rhs_values["even"],
                    "acc": ValueRef.temporary("accumulate_control"),
                },
                results={"dst": ValueRef.temporary("even_accumulator")},
                result_types={"dst": DescriptorResultType()},
            ),
            _op_emit(
                accumulate,
                operands={
                    "acc1": ValueRef.temporary("even_accumulator"),
                    "s1": ValueRef.temporary("lhs_odd"),
                    "s2": rhs_values["odd"],
                    "acc": ValueRef.temporary("accumulate_control"),
                },
                results={"dst": ValueRef.temporary("result_accumulator")},
                result_types={"dst": DescriptorResultType()},
            ),
        )
    )
    result_accumulator_unit = ValueRef.temporary("result_accumulator_unit")
    emits.append(
        EmitRegisterSlice(
            source=ValueRef.temporary("result_accumulator"),
            result=result_accumulator_unit,
            unit_count=1,
        )
    )
    if result_in_accumulator:
        emits.append(
            EmitRegisterCopy(
                source=result_accumulator_unit,
                result=ValueRef.result("result"),
            )
        )
    else:
        emits.append(
            _op_emit(
                move_from_accumulator,
                operands={"src": result_accumulator_unit},
                results={"dst": ValueRef.result("result")},
            )
        )
    return DescriptorRule(
        source_op=vector.vector_dot2f,
        descriptor=accumulate,
        source_nodes=(
            (
                SourceNode.adjacent_definition(
                    "interleave",
                    source_op=vector.vector_interleave,
                    parent_operand=ValueRef.operand("rhs"),
                    node_result=ValueRef.result("result"),
                    guards=(
                        Guard.value_type("even", Vector("bf16", lanes=16)),
                        Guard.value_type("odd", Vector("bf16", lanes=16)),
                        Guard.i64_range("axis", 0, 0),
                    ),
                ),
            )
            if rhs_form == "interleaved"
            else ()
        ),
        priority=1 if rhs_form == "interleaved" else 0,
        guards=guards,
        emit=tuple(emits),
        report_key=report_key,
    )


AIE2P_FLOATING_DOT_RULES = (
    *(
        _vector_dot2f_bf16_rule(
            _BF16X32_VECTOR,
            _F32X16_VECTOR,
            initial_accumulator="zero",
            accumulator_in_accumulator=accumulator_in_accumulator,
            result_in_accumulator=result_in_accumulator,
            rhs_form="interleaved",
            report_key="bf16_dot2_interleaved_rhs_zero",
        )
        for accumulator_in_accumulator in (False, True)
        for result_in_accumulator in (False, True)
    ),
    *(
        _vector_dot2f_bf16_rule(
            _BF16X32_VECTOR,
            _F32X16_VECTOR,
            initial_accumulator="source",
            accumulator_in_accumulator=accumulator_in_accumulator,
            result_in_accumulator=result_in_accumulator,
            rhs_form="interleaved",
            report_key="bf16_dot2_interleaved_rhs",
        )
        for accumulator_in_accumulator in (False, True)
        for result_in_accumulator in (False, True)
    ),
    *(
        _vector_dot2f_bf16_rule(
            _BF16_DOT2_VECTOR,
            _F32_VECTOR,
            initial_accumulator="zero",
            accumulator_in_accumulator=accumulator_in_accumulator,
            result_in_accumulator=result_in_accumulator,
            report_key="bf16_dot2_zero",
        )
        for accumulator_in_accumulator in (False, True)
        for result_in_accumulator in (False, True)
    ),
    *(
        _vector_dot2f_bf16_rule(
            _BF16_DOT2_VECTOR,
            _F32_VECTOR,
            initial_accumulator="source",
            accumulator_in_accumulator=accumulator_in_accumulator,
            result_in_accumulator=result_in_accumulator,
            report_key="bf16_dot2",
        )
        for accumulator_in_accumulator in (False, True)
        for result_in_accumulator in (False, True)
    ),
)
