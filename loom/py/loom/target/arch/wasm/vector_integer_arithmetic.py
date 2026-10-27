# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Packed Wasm SIMD128 integer arithmetic contract rules."""

from collections.abc import Callable

from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.wasm.descriptors import (
    WASM_INTEGER_ARITHMETIC_INSTRUCTIONS,
    WasmIntegerArithmeticInstruction,
)
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    TypePattern,
    ValueRef,
    Vector,
)
from loom.target.low_descriptors import Descriptor

_DIRECT_SOURCE_OPS: dict[str, Op] = {
    "abs": vector.vector_absi,
    "neg": vector.vector_negi,
    "add": vector.vector_addi,
    "sub": vector.vector_subi,
    "mul": vector.vector_muli,
    "min_s": vector.vector_minsi,
    "max_s": vector.vector_maxsi,
    "min_u": vector.vector_minui,
    "max_u": vector.vector_maxui,
}


def _vector_type(element_bit_count: int) -> TypePattern:
    return Vector(
        f"i{element_bit_count}",
        minimum_lanes=1,
        maximum_lanes=128 // element_bit_count,
    )


def _typed_guards(
    type_guard: Callable[[str, TypePattern], Guard],
    fields: tuple[str, ...],
    vector_type: TypePattern,
) -> tuple[Guard, ...]:
    return tuple(type_guard(field, vector_type) for field in fields)


def _direct_rule(
    descriptor_lookup: Callable[[str], Descriptor],
    type_guard: Callable[[str, TypePattern], Guard],
    instruction: WasmIntegerArithmeticInstruction,
) -> DescriptorRule:
    descriptor = descriptor_lookup(f"wasm.{instruction.shape}.{instruction.operation}")
    source_fields = ("input",) if instruction.arity == 1 else ("lhs", "rhs")
    vector_type = _vector_type(instruction.element_bit_count)
    return DescriptorRule(
        source_op=_DIRECT_SOURCE_OPS[instruction.operation],
        descriptor=descriptor,
        guards=_typed_guards(type_guard, (*source_fields, "result"), vector_type),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={field: ValueRef.operand(field) for field in source_fields},
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
        report_key="wasm.integer_arithmetic.native",
    )


def _i8_multiply_emits(
    descriptor_lookup: Callable[[str], Descriptor],
    lhs: ValueRef,
    rhs: ValueRef,
    result: ValueRef,
    *,
    result_is_temporary: bool,
) -> tuple[EmitDescriptorOp, ...]:
    low_multiply = descriptor_lookup("wasm.i16x8.extmul_low_i8x16_u")
    high_multiply = descriptor_lookup("wasm.i16x8.extmul_high_i8x16_u")
    shuffle = descriptor_lookup("wasm.i8x16.shuffle")
    low_product = ValueRef.temporary("low_product")
    high_product = ValueRef.temporary("high_product")
    # Each widening multiply produces eight i16 lanes. Select the low byte of
    # each product and concatenate the low/high halves back into 16 i8 lanes.
    lanes = tuple(range(0, 16, 2)) + tuple(range(16, 32, 2))
    return (
        EmitDescriptorOp(
            descriptor=low_multiply,
            operands={"lhs": lhs, "rhs": rhs},
            results={"dst": low_product},
            result_types={"dst": DescriptorResultType()},
            form=DescriptorEmitForm.OP,
        ),
        EmitDescriptorOp(
            descriptor=high_multiply,
            operands={"lhs": lhs, "rhs": rhs},
            results={"dst": high_product},
            result_types={"dst": DescriptorResultType()},
            form=DescriptorEmitForm.OP,
        ),
        EmitDescriptorOp(
            descriptor=shuffle,
            operands={"lhs": low_product, "rhs": high_product},
            results={"dst": result},
            result_types=(
                {"dst": DescriptorResultType()} if result_is_temporary else {}
            ),
            immediates={f"lane{index}": lane for index, lane in enumerate(lanes)},
            form=DescriptorEmitForm.OP,
        ),
    )


def _i8_multiply_rule(
    descriptor_lookup: Callable[[str], Descriptor],
    type_guard: Callable[[str, TypePattern], Guard],
) -> DescriptorRule:
    emits = _i8_multiply_emits(
        descriptor_lookup,
        ValueRef.operand("lhs"),
        ValueRef.operand("rhs"),
        ValueRef.result("result"),
        result_is_temporary=False,
    )
    vector_type = _vector_type(8)
    return DescriptorRule(
        source_op=vector.vector_muli,
        descriptor=emits[-1].descriptor,
        guards=_typed_guards(type_guard, ("lhs", "rhs", "result"), vector_type),
        emit=emits,
        report_key="wasm.integer_arithmetic.i8_multiply",
    )


def _i64_extrema_rule(
    descriptor_lookup: Callable[[str], Descriptor],
    type_guard: Callable[[str, TypePattern], Guard],
    source_op: Op,
    *,
    is_minimum: bool,
    is_unsigned: bool,
) -> DescriptorRule:
    compare = descriptor_lookup("wasm.i64x2.lt_s")
    bitselect = descriptor_lookup("wasm.v128.bitselect")
    lhs = ValueRef.operand("lhs")
    rhs = ValueRef.operand("rhs")
    compare_lhs = lhs
    compare_rhs = rhs
    emits: list[EmitDescriptorOp] = []
    if is_unsigned:
        sign_mask = ValueRef.temporary("sign_mask")
        compare_lhs = ValueRef.temporary("biased_lhs")
        compare_rhs = ValueRef.temporary("biased_rhs")
        constant = descriptor_lookup("wasm.v128.const")
        bitwise_xor = descriptor_lookup("wasm.v128.xor")
        emits.append(
            EmitDescriptorOp(
                descriptor=constant,
                results={"dst": sign_mask},
                result_types={"dst": DescriptorResultType()},
                immediates={
                    "lo64": 1 << 63,
                    "hi64": 1 << 63,
                },
                form=DescriptorEmitForm.CONST,
            )
        )
        for source, biased in ((lhs, compare_lhs), (rhs, compare_rhs)):
            emits.append(
                EmitDescriptorOp(
                    descriptor=bitwise_xor,
                    operands={"lhs": source, "rhs": sign_mask},
                    results={"dst": biased},
                    result_types={"dst": DescriptorResultType()},
                    form=DescriptorEmitForm.OP,
                )
            )

    condition = ValueRef.temporary("condition")
    emits.append(
        EmitDescriptorOp(
            descriptor=compare,
            operands={"lhs": compare_lhs, "rhs": compare_rhs},
            results={"dst": condition},
            result_types={"dst": DescriptorResultType()},
            form=DescriptorEmitForm.OP,
        )
    )
    emits.append(
        EmitDescriptorOp(
            descriptor=bitselect,
            operands={
                "true_value": lhs if is_minimum else rhs,
                "false_value": rhs if is_minimum else lhs,
                "condition": condition,
            },
            results={"dst": ValueRef.result("result")},
            form=DescriptorEmitForm.OP,
        )
    )
    vector_type = _vector_type(64)
    return DescriptorRule(
        source_op=source_op,
        descriptor=bitselect,
        guards=_typed_guards(type_guard, ("lhs", "rhs", "result"), vector_type),
        emit=tuple(emits),
        report_key="wasm.integer_arithmetic.i64_extrema",
    )


def _fma_rule(
    descriptor_lookup: Callable[[str], Descriptor],
    type_guard: Callable[[str, TypePattern], Guard],
    element_bit_count: int,
    shape: str,
) -> DescriptorRule:
    product = ValueRef.temporary("product")
    if element_bit_count == 8:
        multiply_emits = _i8_multiply_emits(
            descriptor_lookup,
            ValueRef.operand("a"),
            ValueRef.operand("b"),
            product,
            result_is_temporary=True,
        )
    else:
        multiply = descriptor_lookup(f"wasm.{shape}.mul")
        multiply_emits = (
            EmitDescriptorOp(
                descriptor=multiply,
                operands={
                    "lhs": ValueRef.operand("a"),
                    "rhs": ValueRef.operand("b"),
                },
                results={"dst": product},
                result_types={"dst": DescriptorResultType()},
                form=DescriptorEmitForm.OP,
            ),
        )
    add = descriptor_lookup(f"wasm.{shape}.add")
    vector_type = _vector_type(element_bit_count)
    return DescriptorRule(
        source_op=vector.vector_fmai,
        descriptor=add,
        guards=_typed_guards(type_guard, ("a", "b", "c", "result"), vector_type),
        emit=(
            *multiply_emits,
            EmitDescriptorOp(
                descriptor=add,
                operands={"lhs": product, "rhs": ValueRef.operand("c")},
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
        report_key="wasm.integer_arithmetic.fma",
    )


def integer_arithmetic_rules(
    descriptor_lookup: Callable[[str], Descriptor],
    type_guard: Callable[[str, TypePattern], Guard],
) -> tuple[DescriptorRule, ...]:
    """Returns the complete packed integer arithmetic family for Wasm."""

    direct_rules = tuple(
        _direct_rule(descriptor_lookup, type_guard, instruction)
        for instruction in WASM_INTEGER_ARITHMETIC_INSTRUCTIONS
    )
    extrema_rules = tuple(
        _i64_extrema_rule(
            descriptor_lookup,
            type_guard,
            source_op,
            is_minimum=is_minimum,
            is_unsigned=is_unsigned,
        )
        for source_op, is_minimum, is_unsigned in (
            (vector.vector_minsi, True, False),
            (vector.vector_maxsi, False, False),
            (vector.vector_minui, True, True),
            (vector.vector_maxui, False, True),
        )
    )
    fma_rules = tuple(
        _fma_rule(descriptor_lookup, type_guard, element_bit_count, shape)
        for element_bit_count, shape in (
            (8, "i8x16"),
            (16, "i16x8"),
            (32, "i32x4"),
            (64, "i64x2"),
        )
    )
    return (
        *direct_rules,
        _i8_multiply_rule(descriptor_lookup, type_guard),
        *extrema_rules,
        *fma_rules,
    )
