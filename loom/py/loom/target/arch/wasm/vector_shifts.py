# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Packed Wasm SIMD128 integer shift contract rules."""

from collections.abc import Callable, Iterable

from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.wasm.descriptors import WASM_CORE_SIMD128_DESCRIPTOR_SET
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    TypePattern,
    ValueProject,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_I32 = Scalar("i32")
_I64 = Scalar("i64")

_INTEGER_SHIFT_SHAPES = (
    (8, "i8x16", Scalar("i8")),
    (16, "i16x8", Scalar("i16")),
    (32, "i32x4", _I32),
    (64, "i64x2", _I64),
)

_INTEGER_SHIFT_OPERATIONS: tuple[tuple[Op, str], ...] = (
    (vector.vector_shli, "shl"),
    (vector.vector_shrsi, "shr_s"),
    (vector.vector_shrui, "shr_u"),
)


def uniform_shift_rules() -> Iterable[DescriptorRule]:
    constant = descriptor_by_key(WASM_CORE_SIMD128_DESCRIPTOR_SET, "wasm.i32.const")
    wrap = descriptor_by_key(WASM_CORE_SIMD128_DESCRIPTOR_SET, "wasm.i32.wrap_i64")
    for bit_count, shape_name, scalar_type in _INTEGER_SHIFT_SHAPES:
        vector_type = Vector(
            f"i{bit_count}", minimum_lanes=1, maximum_lanes=128 // bit_count
        )
        origin = ValueRef.uniform_element_origin_operand("rhs")
        count = ValueRef.temporary("count")
        runtime_setup = (
            (
                EmitDescriptorOp(
                    descriptor=wrap,
                    operands={"input": origin},
                    results={"dst": count},
                    result_types={"dst": Scalar("i32")},
                ),
            )
            if bit_count == 64
            else ()
        )
        for source_op, operation in _INTEGER_SHIFT_OPERATIONS:
            descriptor = descriptor_by_key(
                WASM_CORE_SIMD128_DESCRIPTOR_SET,
                f"wasm.{shape_name}.{operation}",
            )
            for count_guard, count_value, setup, priority in (
                (
                    Guard.value_exact_i64("rhs"),
                    count,
                    (
                        EmitDescriptorOp(
                            descriptor=constant,
                            results={"dst": count},
                            result_types={"dst": Scalar("i32")},
                            immediates={"i32_value": ValueProject.exact_i64("rhs")},
                            form=DescriptorEmitForm.CONST,
                        ),
                    ),
                    1,
                ),
                (
                    Guard.uniform_element_origin_type("rhs", scalar_type),
                    count if runtime_setup else origin,
                    runtime_setup,
                    2,
                ),
            ):
                yield DescriptorRule(
                    source_op=source_op,
                    descriptor=descriptor,
                    guards=(
                        # Reject varying counts before testing packet shapes.
                        count_guard,
                        *(
                            Guard.value_type(field, vector_type)
                            for field in ("lhs", "rhs", "result")
                        ),
                        # SIMD masks counts to the element width, while narrow
                        # scalar legalization uses an I32 carrier. Restrict the
                        # native selection to their common valid count domain.
                        Guard.value_i64_range("rhs", 0, bit_count - 1),
                    ),
                    emit=(
                        *setup,
                        EmitDescriptorOp(
                            descriptor=descriptor,
                            operands={
                                "value": ValueRef.operand("lhs"),
                                "count": count_value,
                            },
                            results={"dst": ValueRef.result("result")},
                            form=DescriptorEmitForm.OP,
                        ),
                    ),
                    priority=priority,
                    report_key="wasm.integer_shift.uniform.native",
                )


def _repeated_lane_word(element_bit_count: int, lane_value: int) -> int:
    """Packs one integer lane value across a 64-bit v128.const word."""

    lane_mask = (1 << element_bit_count) - 1
    lane_bits = lane_value & lane_mask
    word = 0
    for bit_offset in range(0, 64, element_bit_count):
        word |= lane_bits << bit_offset
    return word


def _typed_guards(
    type_guard: Callable[[str, TypePattern], Guard],
    vector_type: TypePattern,
) -> tuple[Guard, ...]:
    return tuple(type_guard(field, vector_type) for field in ("lhs", "rhs", "result"))


def _varying_shift_rule(
    descriptor_lookup: Callable[[str], Descriptor],
    type_guard: Callable[[str, TypePattern], Guard],
    source_op: Op,
    operation: str,
    vector_type: TypePattern,
    shape_name: str,
    element_bit_count: int,
) -> DescriptorRule:
    native_shift = descriptor_lookup(f"wasm.{shape_name}.{operation}")
    v128_const = descriptor_lookup("wasm.v128.const")
    v128_and = descriptor_lookup("wasm.v128.and")
    lane_equal = descriptor_lookup(f"wasm.{shape_name}.eq")
    i32_const = descriptor_lookup("wasm.i32.const")
    bitselect = descriptor_lookup("wasm.v128.bitselect")

    setup_emits: list[EmitDescriptorOp] = []
    sequence_emits: list[EmitDescriptorOp] = []
    current = ValueRef.operand("lhs")
    stage_count = element_bit_count.bit_length() - 1
    for stage in range(stage_count):
        shift_amount = 1 << stage
        lane_mask_word = _repeated_lane_word(element_bit_count, shift_amount)
        mask = ValueRef.temporary(f"stage{stage}_mask")
        selected_bit = ValueRef.temporary(f"stage{stage}_selected_bit")
        active_lanes = ValueRef.temporary(f"stage{stage}_active_lanes")
        scalar_count = ValueRef.temporary(f"stage{stage}_scalar_count")
        candidate = ValueRef.temporary(f"stage{stage}_candidate")
        next_value = (
            ValueRef.result("result")
            if stage + 1 == stage_count
            else ValueRef.temporary(f"stage{stage}_result")
        )
        setup_emits.extend(
            (
                EmitDescriptorOp(
                    descriptor=v128_const,
                    results={"dst": mask},
                    result_types={"dst": DescriptorResultType()},
                    immediates={"lo64": lane_mask_word, "hi64": lane_mask_word},
                    form=DescriptorEmitForm.CONST,
                ),
                EmitDescriptorOp(
                    descriptor=i32_const,
                    results={"dst": scalar_count},
                    result_types={"dst": _I32},
                    immediates={"i32_value": shift_amount},
                    form=DescriptorEmitForm.CONST,
                ),
            )
        )
        sequence_emits.extend(
            (
                EmitDescriptorOp(
                    descriptor=v128_and,
                    operands={"lhs": ValueRef.operand("rhs"), "rhs": mask},
                    results={"dst": selected_bit},
                    result_types={"dst": DescriptorResultType()},
                    form=DescriptorEmitForm.OP,
                ),
                EmitDescriptorOp(
                    descriptor=lane_equal,
                    operands={"lhs": selected_bit, "rhs": mask},
                    results={"dst": active_lanes},
                    result_types={"dst": DescriptorResultType()},
                    form=DescriptorEmitForm.OP,
                ),
                EmitDescriptorOp(
                    descriptor=native_shift,
                    operands={"value": current, "count": scalar_count},
                    results={"dst": candidate},
                    result_types={"dst": DescriptorResultType()},
                    form=DescriptorEmitForm.OP,
                ),
                EmitDescriptorOp(
                    descriptor=bitselect,
                    operands={
                        "true_value": candidate,
                        "false_value": current,
                        "condition": active_lanes,
                    },
                    results={"dst": next_value},
                    result_types=(
                        None
                        if next_value == ValueRef.result("result")
                        else {"dst": DescriptorResultType()}
                    ),
                    form=DescriptorEmitForm.OP,
                ),
            )
        )
        current = next_value

    return DescriptorRule(
        source_op=source_op,
        descriptor=bitselect,
        guards=_typed_guards(type_guard, vector_type),
        emit=(*setup_emits, *sequence_emits),
        report_key="wasm.integer_shift.varying.barrel",
    )


def varying_shift_rules(
    descriptor_lookup: Callable[[str], Descriptor],
    type_guard: Callable[[str, TypePattern], Guard],
) -> tuple[DescriptorRule, ...]:
    """Returns packed logarithmic barrel shifts for every integer lane width."""

    return tuple(
        _varying_shift_rule(
            descriptor_lookup,
            type_guard,
            source_op,
            operation,
            Vector(
                f"i{element_bit_count}",
                minimum_lanes=1,
                maximum_lanes=128 // element_bit_count,
            ),
            shape_name,
            element_bit_count,
        )
        for element_bit_count, shape_name, _ in _INTEGER_SHIFT_SHAPES
        for source_op, operation in _INTEGER_SHIFT_OPERATIONS
    )
