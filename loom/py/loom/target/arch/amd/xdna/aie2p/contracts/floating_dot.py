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
    BF16_CONVERSION_ROUNDING,
    vector_data_path_control,
)
from loom.target.arch.amd.xdna.aie2p.contracts.packet_program import (
    PacketProgram,
    shift_integer_packet_fixed,
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
_F16_DOT2_VECTOR = Vector("f16", minimum_static_elements=2, maximum_static_elements=32)
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


def _convert_f16_to_bf16(
    program: PacketProgram,
    shared_splats: dict[int, ValueRef],
    zero: ValueRef,
    source: ValueRef,
    name: str,
) -> ValueRef:
    """Rounds FP16 lanes to BF16, flushing FP16 subnormals to signed zero."""

    def splat(suffix: str, value: int) -> ValueRef:
        result = shared_splats.get(value)
        if result is None:
            result = program.splat(f"{name}_{suffix}", value)
            shared_splats[value] = result
        return result

    absolute_mask = splat("absolute_mask", 0x7FFF)
    magnitude = program.binary(
        f"{name}_magnitude", "and.bits512", source, absolute_mask
    )
    sign = program.binary(f"{name}_sign", "sub.i16x32", source, magnitude)
    positioned = shift_integer_packet_fixed(
        program,
        f"{name}_positioned",
        magnitude,
        -3,
        rounding_mode=BF16_CONVERSION_ROUNDING,
    )

    normal_bias = splat("normal_bias", 0x3800)
    normal_unsigned = program.binary(
        f"{name}_normal_unsigned", "add.i16x32", positioned, normal_bias
    )
    normal_threshold = splat("normal_threshold", 0x0400)
    below_normal = program.compare_unsigned_less_than(
        f"{name}_below_normal", magnitude, normal_threshold
    )
    finite_unsigned = program.select(
        f"{name}_finite_unsigned", zero, normal_unsigned, below_normal
    )

    special_threshold = splat("special_threshold", 0x7C00)
    is_special = program.compare_unsigned_greater_equal(
        f"{name}_is_special", magnitude, special_threshold
    )
    is_nan = program.compare_unsigned_less_than(
        f"{name}_is_nan", special_threshold, magnitude
    )
    infinity = splat("infinity", 0x7F80)
    quiet_nan = splat("quiet_nan", 0x7FC0)
    special_unsigned = program.select(
        f"{name}_special_unsigned", quiet_nan, infinity, is_nan
    )
    high_unsigned = program.select(
        f"{name}_high_unsigned", special_unsigned, finite_unsigned, is_special
    )
    return program.binary(f"{name}_bf16", "or.bits512", high_unsigned, sign)


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


def _vector_dot2f_f16_rule() -> DescriptorRule:
    """Runs FP16 pair dots through AIE2P's native BF16 arithmetic."""

    config_constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    shuffle = _descriptor("amd.xdna.aie2p.shuffle.x.configured")
    move_to_accumulator = _descriptor(
        "amd.xdna.aie2p.move.vector512.to.accumulator512.low"
    )
    first_accumulate = _descriptor("amd.xdna.aie2p.accumulate.bf16x16.configured")
    accumulate = _descriptor("amd.xdna.aie2p.accumulate.bf16x32.configured")
    move_from_accumulator = _descriptor(
        "amd.xdna.aie2p.move.accumulator512.to.vector512"
    )

    program = PacketProgram(16, "f16_dot_")
    converted: dict[str, ValueRef] = {}
    shared_splats: dict[int, ValueRef] = {}
    lhs = ValueRef.operand("lhs")
    zero = program.binary("zero", "sub.i16x32", lhs, lhs)
    for operand_name in ("lhs", "rhs"):
        source = ValueRef.operand(operand_name)
        converted[operand_name] = _convert_f16_to_bf16(
            program,
            shared_splats,
            zero,
            source,
            operand_name,
        )
    emits: list[ContractEmit] = [*program.emits]

    lane_controls: dict[str, ValueRef] = {}
    for lane_group, control in zip(
        ("even", "odd"), _BF16_DOT2_DEINTERLEAVE_CONTROLS, strict=True
    ):
        control_value = ValueRef.temporary(f"f16_dot_{lane_group}_control")
        emits.append(_constant_emit(config_constant, control_value, control))
        lane_controls[lane_group] = control_value

    accumulator = ValueRef.temporary("f16_dot_initial_accumulator")
    emits.append(
        _op_emit(
            move_to_accumulator,
            operands={"src": ValueRef.operand("acc")},
            results={"dst": accumulator},
            result_types={"dst": DescriptorResultType()},
        )
    )
    accumulate_control = ValueRef.temporary("f16_dot_accumulate_control")
    emits.append(
        _constant_emit(
            config_constant,
            accumulate_control,
            _BF16_ELEMENTWISE_MULTIPLY_CONTROL,
        )
    )

    for lane_index, lane_group in enumerate(("even", "odd")):
        grouped_operands: dict[str, ValueRef] = {}
        for operand_name in ("lhs", "rhs"):
            grouped = ValueRef.temporary(f"f16_dot_{operand_name}_{lane_group}")
            emits.append(
                _op_emit(
                    shuffle,
                    operands={
                        "s1": converted[operand_name],
                        "s2": converted[operand_name],
                        "mod": lane_controls[lane_group],
                    },
                    results={"dst": grouped},
                    result_types={"dst": DescriptorResultType()},
                )
            )
            grouped_operands[operand_name] = grouped
        next_accumulator = ValueRef.temporary(f"f16_dot_accumulator_{lane_group}")
        emits.append(
            _op_emit(
                first_accumulate if lane_index == 0 else accumulate,
                operands={
                    "acc1": accumulator,
                    "s1": grouped_operands["lhs"],
                    "s2": grouped_operands["rhs"],
                    "acc": accumulate_control,
                },
                results={"dst": next_accumulator},
                result_types={"dst": DescriptorResultType()},
            )
        )
        accumulator = next_accumulator

    result_accumulator_unit = ValueRef.temporary("f16_dot_result_accumulator_unit")
    emits.extend(
        (
            EmitRegisterSlice(
                source=accumulator,
                result=result_accumulator_unit,
                unit_count=1,
            ),
            _op_emit(
                move_from_accumulator,
                operands={"src": result_accumulator_unit},
                results={"dst": ValueRef.result("result")},
            ),
        )
    )
    return DescriptorRule(
        source_op=vector.vector_dot2f,
        descriptor=accumulate,
        guards=(
            Guard.value_type("lhs", _F16_DOT2_VECTOR),
            Guard.value_type("rhs", _F16_DOT2_VECTOR),
            Guard.value_type("acc", _F32_VECTOR),
            Guard.value_type("result", _F32_VECTOR),
            Guard.low_value_register_class("acc", "aie2p.vec256"),
            Guard.low_value_register_class("result", "aie2p.vec256"),
        ),
        emit=tuple(emits),
        report_key="f16_dot2_bf16_round_nearest",
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
    _vector_dot2f_f16_rule(),
)
