# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Straight-line i32x4 descriptor programs for Wasm SIMD128."""

from collections.abc import Callable, Mapping

from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    EmitDescriptorOp,
    Scalar,
    SourceValueKind,
    ValueRef,
)
from loom.target.low_descriptors import Descriptor

_I32 = Scalar("i32")


def _repeated_i32_word(value: int) -> int:
    lane_bits = value & 0xFFFFFFFF
    return lane_bits | (lane_bits << 32)


class PackedI32Program:
    """Builds one straight-line i32x4 descriptor program."""

    def __init__(self, descriptor_lookup: Callable[[str], Descriptor]) -> None:
        self._descriptor_lookup = descriptor_lookup
        self.emits: list[EmitDescriptorOp] = []
        self._vector_constants: dict[int, ValueRef] = {}
        self._vector_constant_values: dict[ValueRef, int] = {}
        self._scalar_constants: dict[int, ValueRef] = {}

    @staticmethod
    def _result(result_name: str | None) -> ValueRef:
        return (
            ValueRef.result("result")
            if result_name is None
            else ValueRef.temporary(result_name)
        )

    def constant(self, result_name: str, value: int) -> ValueRef:
        if existing := self._vector_constants.get(value):
            return existing
        result = self._result(result_name)
        word = _repeated_i32_word(value)
        self.emits.append(
            EmitDescriptorOp(
                descriptor=self._descriptor_lookup("wasm.v128.const"),
                results={"dst": result},
                result_types={"dst": DescriptorResultType()},
                immediates={"lo64": word, "hi64": word},
                form=DescriptorEmitForm.CONST,
            )
        )
        self._vector_constants[value] = result
        self._vector_constant_values[result] = value
        return result

    def constant_value(self, value: ValueRef) -> int:
        """Returns the literal repeated in every lane of a vector constant."""

        try:
            return self._vector_constant_values[value]
        except KeyError as exc:
            raise ValueError(
                "packed shift counts must be compile-time constants"
            ) from exc

    def scalar_constant(self, result_name: str, value: int) -> ValueRef:
        if existing := self._scalar_constants.get(value):
            return existing
        result = self._result(result_name)
        self.emits.append(
            EmitDescriptorOp(
                descriptor=self._descriptor_lookup("wasm.i32.const"),
                results={"dst": result},
                result_types={"dst": _I32},
                immediates={"i32_value": value},
                form=DescriptorEmitForm.CONST,
            )
        )
        self._scalar_constants[value] = result
        return result

    def operation(
        self,
        result_name: str | None,
        descriptor_key: str,
        operands: Mapping[str, ValueRef],
    ) -> ValueRef:
        result = self._result(result_name)
        return self.operation_to(result, descriptor_key, operands)

    def operation_to(
        self,
        result: ValueRef,
        descriptor_key: str,
        operands: Mapping[str, ValueRef],
    ) -> ValueRef:
        self.emits.append(
            EmitDescriptorOp(
                descriptor=self._descriptor_lookup(descriptor_key),
                operands=operands,
                results={"dst": result},
                result_types=(
                    {"dst": DescriptorResultType()}
                    if result.kind is SourceValueKind.TEMPORARY
                    else None
                ),
                form=DescriptorEmitForm.OP,
            )
        )
        return result

    def binary(
        self,
        result_name: str | None,
        descriptor_key: str,
        lhs: ValueRef,
        rhs: ValueRef,
    ) -> ValueRef:
        return self.operation(
            result_name,
            descriptor_key,
            {"lhs": lhs, "rhs": rhs},
        )

    def shift(
        self,
        result_name: str | None,
        descriptor_key: str,
        value: ValueRef,
        count: ValueRef,
    ) -> ValueRef:
        return self.operation(
            result_name,
            descriptor_key,
            {"value": value, "count": count},
        )

    def select(
        self,
        result_name: str | None,
        true_value: ValueRef,
        false_value: ValueRef,
        condition: ValueRef,
    ) -> ValueRef:
        return self.operation(
            result_name,
            "wasm.v128.bitselect",
            {
                "true_value": true_value,
                "false_value": false_value,
                "condition": condition,
            },
        )
