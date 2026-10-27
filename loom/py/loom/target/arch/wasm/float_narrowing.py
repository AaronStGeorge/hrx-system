# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact scalar and SIMD float-narrowing Wasm contract rules."""

from collections.abc import Callable

from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.vector import defs as vector
from loom.target.arch.wasm.packed_i32 import PackedI32Program
from loom.target.contracts import (
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    TypePattern,
    ValueRef,
    Vector,
)
from loom.target.emit.float_narrowing import (
    BF16_FORMAT,
    BF16_SOURCE_FORMAT,
    F8E4M3_FORMAT,
    F8E5M2_FORMAT,
    F16_FORMAT,
    F16_SOURCE_FORMAT,
    F32_FORMAT,
    F64_FORMAT,
    BinaryFloatFormat,
    FloatNarrowingDescriptors,
    IntegerNarrowingDescriptors,
    NarrowFloatFormat,
    NarrowFloatSubnormalRounding,
    build_f32_bits_to_bf16_with_recipe,
    build_f32_to_bf16_emits,
    build_float_bits_to_narrow_float_with_recipe,
    build_float_to_narrow_float_emits,
    build_integer_bits_to_narrow_float_emits,
    build_representable_f32_to_bf16_emits,
)
from loom.target.low_descriptors import Descriptor

_F8E4M3 = Scalar("f8E4M3")
_F8E5M2 = Scalar("f8E5M2")
_F16 = Scalar("f16")
_BF16 = Scalar("bf16")
_F32 = Scalar("f32")
_F64 = Scalar("f64")
_F32_PACKET = Vector("f32", minimum_lanes=1, maximum_lanes=4)


class _PackedNarrowingRecipe(PackedI32Program):
    """Builds exact narrowing payloads in four parallel i32 lanes."""

    def __init__(self, descriptor_lookup: Callable[[str], Descriptor]) -> None:
        super().__init__(descriptor_lookup)
        self.descriptors = IntegerNarrowingDescriptors(
            integer_bit_width=32,
            integer_constant=descriptor_lookup("wasm.v128.const"),
            integer_add=descriptor_lookup("wasm.i32x4.add"),
            integer_subtract=descriptor_lookup("wasm.i32x4.sub"),
            integer_shift_left=descriptor_lookup("wasm.i32x4.shl"),
            integer_shift_right_logical=descriptor_lookup("wasm.i32x4.shr_u"),
            integer_bitwise_and=descriptor_lookup("wasm.v128.and"),
            integer_bitwise_or=descriptor_lookup("wasm.v128.or"),
            integer_less_than_nonnegative=descriptor_lookup("wasm.i32x4.lt_u"),
            integer_greater_than_equal_nonnegative=descriptor_lookup("wasm.i32x4.ge_u"),
            integer_greater_than_nonnegative=descriptor_lookup("wasm.i32x4.gt_u"),
            integer_select=descriptor_lookup("wasm.v128.bitselect"),
        )

    def finish(self) -> tuple[EmitDescriptorOp, ...]:
        return tuple(self.emits)

    def integer_constant(self, result_name: str, value: int) -> ValueRef:
        return self.constant(result_name, value)

    def float_constant(
        self, result_name: str, bits: int, descriptor: Descriptor
    ) -> ValueRef:
        if descriptor.key != "wasm.v128.const":
            raise ValueError("packed float constants require wasm.v128.const")
        return self.constant(result_name, bits)

    def _integer_operands(
        self,
        result_name: str,
        descriptor: Descriptor,
        lhs: ValueRef,
        rhs: ValueRef,
    ) -> dict[str, ValueRef]:
        if descriptor.key not in (
            self.descriptors.integer_shift_left.key,
            self.descriptors.integer_shift_right_logical.key,
        ):
            return {"lhs": lhs, "rhs": rhs}
        count_value = self.constant_value(rhs)
        count = self.scalar_constant(f"{result_name}_count", count_value)
        return {"value": lhs, "count": count}

    def integer_binary(
        self,
        result_name: str,
        descriptor: Descriptor,
        lhs: ValueRef,
        rhs: ValueRef,
    ) -> ValueRef:
        return self.operation(
            result_name,
            descriptor.key,
            self._integer_operands(result_name, descriptor, lhs, rhs),
        )

    def integer_binary_to(
        self,
        result: ValueRef,
        descriptor: Descriptor,
        lhs: ValueRef,
        rhs: ValueRef,
    ) -> ValueRef:
        return self.operation_to(
            result,
            descriptor.key,
            self._integer_operands(result.field, descriptor, lhs, rhs),
        )

    def float_add(
        self,
        result_name: str,
        descriptor: Descriptor,
        lhs: ValueRef,
        rhs: ValueRef,
    ) -> ValueRef:
        return self.binary(result_name, descriptor.key, lhs, rhs)

    @staticmethod
    def reinterpret_float_as_integer(
        result_name: str, descriptor: Descriptor | None, value: ValueRef
    ) -> ValueRef:
        if descriptor is not None:
            raise ValueError(
                "packed float and integer lanes must alias one v128 carrier"
            )
        return value

    @staticmethod
    def reinterpret_integer_as_float(
        result_name: str, descriptor: Descriptor | None, value: ValueRef
    ) -> ValueRef:
        if descriptor is not None:
            raise ValueError(
                "packed integer and float lanes must alias one v128 carrier"
            )
        return value

    def integer_select(
        self,
        result_name: str,
        true_value: ValueRef,
        false_value: ValueRef,
        condition: ValueRef,
    ) -> ValueRef:
        return self.select(
            result_name,
            true_value,
            false_value,
            condition,
        )

    def integer_select_to(
        self,
        result: ValueRef,
        true_value: ValueRef,
        false_value: ValueRef,
        condition: ValueRef,
    ) -> ValueRef:
        return self.operation_to(
            result,
            "wasm.v128.bitselect",
            {
                "true_value": true_value,
                "false_value": false_value,
                "condition": condition,
            },
        )

    def pack(
        self,
        payloads: ValueRef,
        result: ValueRef,
        element_bit_count: int,
    ) -> None:
        zero = self.constant("pack_zero", 0)
        packed_i16 = self.operation_to(
            result if element_bit_count == 16 else ValueRef.temporary("packed_i16"),
            "wasm.i16x8.narrow_i32x4_u",
            {"low": payloads, "high": zero},
        )
        if element_bit_count == 8:
            self.operation_to(
                result,
                "wasm.i8x16.narrow_i16x8_u",
                {"low": packed_i16, "high": zero},
            )
        elif element_bit_count != 16:
            raise ValueError("packed float narrowing requires 8- or 16-bit results")


def _integer_narrowing_descriptors(
    descriptor_lookup: Callable[[str], Descriptor],
    integer_bit_width: int,
) -> IntegerNarrowingDescriptors:
    integer_name = f"i{integer_bit_width}"
    return IntegerNarrowingDescriptors(
        integer_bit_width=integer_bit_width,
        integer_constant=descriptor_lookup(f"wasm.{integer_name}.const"),
        integer_add=descriptor_lookup(f"wasm.{integer_name}.add"),
        integer_subtract=descriptor_lookup(f"wasm.{integer_name}.sub"),
        integer_shift_left=descriptor_lookup(f"wasm.{integer_name}.shl"),
        integer_shift_right_logical=descriptor_lookup(f"wasm.{integer_name}.shr_u"),
        integer_bitwise_and=descriptor_lookup(f"wasm.{integer_name}.and"),
        integer_bitwise_or=descriptor_lookup(f"wasm.{integer_name}.or"),
        integer_less_than_nonnegative=descriptor_lookup(f"wasm.{integer_name}.lt_u"),
        integer_greater_than_equal_nonnegative=descriptor_lookup(
            f"wasm.{integer_name}.ge_u"
        ),
        integer_greater_than_nonnegative=descriptor_lookup(f"wasm.{integer_name}.gt_u"),
        integer_select=descriptor_lookup(f"wasm.{integer_name}.select"),
    )


def _f32_narrowing_descriptors(
    descriptor_lookup: Callable[[str], Descriptor],
) -> FloatNarrowingDescriptors:
    return FloatNarrowingDescriptors(
        integer=_integer_narrowing_descriptors(descriptor_lookup, 32),
        float_constant=descriptor_lookup("wasm.f32.const"),
        float_add=descriptor_lookup("wasm.f32.add"),
        reinterpret_float_as_integer=descriptor_lookup("wasm.i32.reinterpret_f32"),
        reinterpret_integer_as_float=descriptor_lookup("wasm.f32.reinterpret_i32"),
    )


def _f32_to_bf16_rule(
    descriptors: FloatNarrowingDescriptors,
    type_guard: Callable[[str, TypePattern], Guard],
) -> DescriptorRule:
    emits = build_f32_to_bf16_emits(
        descriptors,
        ValueRef.operand("input"),
        ValueRef.result("result"),
    )
    return DescriptorRule(
        source_op=scalar_conversion.scalar_fptrunc,
        descriptor=emits[-1].descriptor,
        guards=(
            type_guard("input", _F32),
            type_guard("result", _BF16),
        ),
        emit=emits,
        report_key="exact_binary32_to_bfloat16",
    )


def _representable_f32_to_bf16_rule(
    descriptors: FloatNarrowingDescriptors,
    type_guard: Callable[[str, TypePattern], Guard],
    *,
    preserve_nan: bool,
) -> DescriptorRule:
    emits = build_representable_f32_to_bf16_emits(
        descriptors,
        ValueRef.operand("input"),
        ValueRef.result("result"),
        preserve_nan=preserve_nan,
    )
    return DescriptorRule(
        source_op=scalar_conversion.scalar_fptrunc,
        descriptor=emits[-1].descriptor,
        guards=(
            type_guard("input", _F32),
            type_guard("result", _BF16),
            *((Guard.value_not_nan("input"),) if not preserve_nan else ()),
            Guard.value_float_exactly_representable_as("input", "result"),
        ),
        emit=emits,
        report_key=(
            "exactly_representable_binary32_to_bfloat16"
            + ("_preserve_nan" if preserve_nan else "")
        ),
    )


def _f32_to_narrow_float_rule(
    descriptors: FloatNarrowingDescriptors,
    type_guard: Callable[[str, TypePattern], Guard],
    result_type: TypePattern,
    narrow_format: NarrowFloatFormat,
    report_key: str,
) -> DescriptorRule:
    emits = build_float_to_narrow_float_emits(
        descriptors,
        F32_FORMAT,
        narrow_format,
        ValueRef.operand("input"),
        ValueRef.result("result"),
        subnormal_rounding=NarrowFloatSubnormalRounding.RNE_FLOAT_ADD,
    )
    return DescriptorRule(
        source_op=scalar_conversion.scalar_fptrunc,
        descriptor=emits[-1].descriptor,
        guards=(
            type_guard("input", _F32),
            type_guard("result", result_type),
        ),
        emit=emits,
        report_key=report_key,
    )


def _f64_to_narrow_float_rule(
    descriptor_lookup: Callable[[str], Descriptor],
    descriptors: IntegerNarrowingDescriptors,
    type_guard: Callable[[str, TypePattern], Guard],
    result_type: TypePattern,
    narrow_format: NarrowFloatFormat,
    result_name: str,
) -> DescriptorRule:
    reinterpret_descriptor = descriptor_lookup("wasm.i64.reinterpret_f64")
    emits = [
        EmitDescriptorOp(
            descriptor=reinterpret_descriptor,
            operands={"input": ValueRef.operand("input")},
            results={"dst": ValueRef.temporary("input_bits")},
            result_types={"dst": DescriptorResultType()},
        )
    ]
    emits.extend(
        build_integer_bits_to_narrow_float_emits(
            descriptors,
            F64_FORMAT,
            narrow_format,
            ValueRef.temporary("input_bits"),
            ValueRef.temporary("carrier"),
            subnormal_rounding=NarrowFloatSubnormalRounding.INTEGER,
        )
    )
    wrap_descriptor = descriptor_lookup("wasm.i32.wrap_i64")
    emits.append(
        EmitDescriptorOp(
            descriptor=wrap_descriptor,
            operands={"input": ValueRef.temporary("carrier")},
            results={"dst": ValueRef.result("result")},
        )
    )
    return DescriptorRule(
        source_op=scalar_conversion.scalar_fptrunc,
        descriptor=wrap_descriptor,
        guards=(
            type_guard("input", _F64),
            type_guard("result", result_type),
        ),
        emit=tuple(emits),
        report_key=f"exact_binary64_to_{result_name}",
    )


def _integer_carrier_to_float8_rule(
    descriptors: IntegerNarrowingDescriptors,
    type_guard: Callable[[str, TypePattern], Guard],
    source_type: TypePattern,
    source_format: BinaryFloatFormat,
    source_name: str,
    result_type: TypePattern,
    narrow_format: NarrowFloatFormat,
    result_name: str,
) -> DescriptorRule:
    if (
        source_format.exponent_bias
        >= narrow_format.exponent_bias + narrow_format.mantissa_bits + 1
    ):
        subnormal_rounding = NarrowFloatSubnormalRounding.INTEGER
    else:
        subnormal_rounding = NarrowFloatSubnormalRounding.INTEGER_SOURCE_SUBNORMALS
    emits = build_integer_bits_to_narrow_float_emits(
        descriptors,
        source_format,
        narrow_format,
        ValueRef.operand("input"),
        ValueRef.result("result"),
        subnormal_rounding=subnormal_rounding,
    )
    return DescriptorRule(
        source_op=scalar_conversion.scalar_fptrunc,
        descriptor=emits[-1].descriptor,
        guards=(
            type_guard("input", source_type),
            type_guard("result", result_type),
        ),
        emit=emits,
        report_key=f"exact_{source_name}_to_{result_name}",
    )


def _f32_packet_to_narrow_float_rule(
    descriptor_lookup: Callable[[str], Descriptor],
    type_guard: Callable[[str, TypePattern], Guard],
    result_element: str,
    narrow_format: NarrowFloatFormat,
) -> DescriptorRule:
    recipe = _PackedNarrowingRecipe(descriptor_lookup)
    payloads = ValueRef.temporary("narrow_payloads")
    float_descriptors = FloatNarrowingDescriptors(
        integer=recipe.descriptors,
        float_constant=descriptor_lookup("wasm.v128.const"),
        float_add=descriptor_lookup("wasm.f32x4.add"),
        reinterpret_float_as_integer=None,
        reinterpret_integer_as_float=None,
    )
    if result_element == "bf16":
        build_f32_bits_to_bf16_with_recipe(
            recipe,
            ValueRef.operand("input"),
            payloads,
            exact_finite_values=False,
            preserve_nan=True,
        )
    else:
        build_float_bits_to_narrow_float_with_recipe(
            recipe,
            F32_FORMAT,
            narrow_format,
            ValueRef.operand("input"),
            payloads,
            subnormal_rounding=NarrowFloatSubnormalRounding.RNE_FLOAT_ADD,
            preserve_nan=True,
            float_descriptors=float_descriptors,
        )
    recipe.pack(payloads, ValueRef.result("result"), narrow_format.bit_width)
    result_type = Vector(result_element, minimum_lanes=1, maximum_lanes=4)
    return DescriptorRule(
        source_op=vector.vector_fptrunc,
        descriptor=recipe.emits[-1].descriptor,
        guards=(
            type_guard("input", _F32_PACKET),
            type_guard("result", result_type),
            Guard.value_static_element_count_eq("input", "result"),
        ),
        emit=recipe.finish(),
        report_key=f"wasm.float_narrowing.exact_f32_to_{result_element.lower()}",
    )


def float_narrowing_rules(
    descriptor_lookup: Callable[[str], Descriptor],
    type_guard: Callable[[str, TypePattern], Guard],
) -> tuple[DescriptorRule, ...]:
    """Returns the complete exact scalar and SIMD float-narrowing matrix."""

    f32_descriptors = _f32_narrowing_descriptors(descriptor_lookup)
    i32_descriptors = f32_descriptors.integer
    i64_descriptors = _integer_narrowing_descriptors(descriptor_lookup, 64)
    rules = [
        _representable_f32_to_bf16_rule(
            f32_descriptors, type_guard, preserve_nan=False
        ),
        _representable_f32_to_bf16_rule(f32_descriptors, type_guard, preserve_nan=True),
        _f32_to_bf16_rule(f32_descriptors, type_guard),
        _f32_to_narrow_float_rule(
            f32_descriptors,
            type_guard,
            _F16,
            F16_FORMAT,
            "exact_binary32_to_f16",
        ),
        _f32_to_narrow_float_rule(
            f32_descriptors,
            type_guard,
            _F8E4M3,
            F8E4M3_FORMAT,
            "exact_binary32_to_f8e4m3",
        ),
        _f32_to_narrow_float_rule(
            f32_descriptors,
            type_guard,
            _F8E5M2,
            F8E5M2_FORMAT,
            "exact_binary32_to_f8e5m2",
        ),
    ]
    rules.extend(
        _f64_to_narrow_float_rule(
            descriptor_lookup,
            i64_descriptors,
            type_guard,
            result_type,
            narrow_format,
            result_name,
        )
        for result_type, narrow_format, result_name in (
            (_F16, F16_FORMAT, "f16"),
            (_BF16, BF16_FORMAT, "bfloat16"),
            (_F8E4M3, F8E4M3_FORMAT, "f8e4m3"),
            (_F8E5M2, F8E5M2_FORMAT, "f8e5m2"),
        )
    )
    rules.extend(
        _integer_carrier_to_float8_rule(
            i32_descriptors,
            type_guard,
            source_type,
            source_format,
            source_name,
            result_type,
            narrow_format,
            result_name,
        )
        for source_type, source_format, source_name in (
            (_F16, F16_SOURCE_FORMAT, "binary16"),
            (_BF16, BF16_SOURCE_FORMAT, "bfloat16"),
        )
        for result_type, narrow_format, result_name in (
            (_F8E4M3, F8E4M3_FORMAT, "f8e4m3"),
            (_F8E5M2, F8E5M2_FORMAT, "f8e5m2"),
        )
    )
    rules.extend(
        _f32_packet_to_narrow_float_rule(
            descriptor_lookup,
            type_guard,
            result_element,
            narrow_format,
        )
        for result_element, narrow_format in (
            ("f8E4M3", F8E4M3_FORMAT),
            ("f8E5M2", F8E5M2_FORMAT),
            ("f16", F16_FORMAT),
            ("bf16", BF16_FORMAT),
        )
    )
    return tuple(rules)
