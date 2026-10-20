# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AIE2P binary32 arithmetic through the native accumulator file."""

from __future__ import annotations

from collections.abc import Mapping

from loom.target.arch.amd.xdna.aie2p.contracts.data_path import (
    F32_ACCUMULATOR_ADD_CONTROL,
)
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    ContractEmit,
    DescriptorEmitForm,
    DescriptorResultType,
    EmitDescriptorOp,
    EmitRegisterCopy,
    EmitRegisterSlice,
    ResultTypeBinding,
    ValueRef,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor


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


class F32AccumulatorProgram:
    """Builds binary32 arithmetic while retaining accumulator operands."""

    def __init__(self, temporary_prefix: str = "") -> None:
        self.emits: list[ContractEmit] = []
        self.temporary_prefix = temporary_prefix
        self._arithmetic_control: ValueRef | None = None

    def temporary(self, name: str) -> ValueRef:
        return ValueRef.temporary(f"{self.temporary_prefix}{name}")

    def _control(self) -> ValueRef:
        control = self._arithmetic_control
        if control is not None:
            return control
        control = self.temporary("arithmetic_control")
        self.emits.append(
            _constant_emit(
                _descriptor("amd.xdna.aie2p.constant.i32.mova"),
                control,
                F32_ACCUMULATOR_ADD_CONTROL,
            )
        )
        self._arithmetic_control = control
        return control

    def vector_to_accumulator(self, name: str, source: ValueRef) -> ValueRef:
        """Moves one X-carried F32 packet into the low accumulator quarter."""

        accumulator = self.temporary(f"{name}_accumulator")
        self.emits.append(
            _op_emit(
                _descriptor("amd.xdna.aie2p.move.vector512.to.accumulator512.low"),
                operands={"src": source},
                results={"dst": accumulator},
                result_types={"dst": DescriptorResultType()},
            )
        )
        return accumulator

    def binary(
        self,
        name: str,
        lhs: ValueRef,
        rhs: ValueRef,
        operation_descriptor_key: str,
    ) -> ValueRef:
        """Emits one configured F32 accumulator add or subtract."""

        result = self.temporary(f"{name}_accumulator")
        self.emits.append(
            _op_emit(
                _descriptor(operation_descriptor_key),
                operands={"acc1": lhs, "acc2": rhs, "acc": self._control()},
                results={"dst": result},
                result_types={"dst": DescriptorResultType()},
            )
        )
        return result

    def accumulator_to_vector(
        self,
        name: str,
        source: ValueRef,
        result: ValueRef,
        *,
        bind_result_type: bool = False,
    ) -> None:
        """Moves the low sixteen F32 lanes of an accumulator into an X carrier."""

        accumulator_unit = self.temporary(f"{name}_accumulator_unit")
        self.emits.extend(
            (
                EmitRegisterSlice(
                    source=source,
                    result=accumulator_unit,
                    unit_count=1,
                ),
                _op_emit(
                    _descriptor("amd.xdna.aie2p.move.accumulator512.to.vector512"),
                    operands={"src": accumulator_unit},
                    results={"dst": result},
                    result_types=(
                        {"dst": DescriptorResultType()} if bind_result_type else None
                    ),
                ),
            )
        )


def emit_f32_vector_accumulator_binary(
    lhs: ValueRef,
    rhs: ValueRef,
    result: ValueRef,
    operation_descriptor_key: str,
    *,
    extract_scalar_result: bool,
    temporary_prefix: str = "",
) -> tuple[ContractEmit, ...]:
    """Builds one vector or broadcast-scalar F32 accumulator operation."""

    program = F32AccumulatorProgram(temporary_prefix)
    lhs_accumulator = program.vector_to_accumulator("lhs", lhs)
    rhs_accumulator = program.vector_to_accumulator("rhs", rhs)
    result_accumulator = program.binary(
        "result", lhs_accumulator, rhs_accumulator, operation_descriptor_key
    )
    final_vector = (
        program.temporary("result_vector") if extract_scalar_result else result
    )
    program.accumulator_to_vector(
        "result",
        result_accumulator,
        final_vector,
        bind_result_type=extract_scalar_result,
    )
    if extract_scalar_result:
        program.emits.append(
            EmitDescriptorOp(
                descriptor=_descriptor("amd.xdna.aie2p.extract.i32.immediate"),
                operands={"s1": final_vector},
                results={"dst": result},
                result_types={"dst": DescriptorResultType()},
                immediates={"idx": 0},
                form=DescriptorEmitForm.OP,
            )
        )
    return tuple(program.emits)


def emit_f32_planned_accumulator_binary(
    lhs: ValueRef,
    rhs: ValueRef,
    result: ValueRef,
    operation_descriptor_key: str,
    *,
    lhs_in_accumulator: bool,
    rhs_in_accumulator: bool,
    result_in_accumulator: bool,
    temporary_prefix: str = "",
) -> tuple[ContractEmit, ...]:
    """Builds one F32 packet operation for planned X/accumulator carriers."""

    program = F32AccumulatorProgram(temporary_prefix)
    lhs_accumulator = (
        lhs if lhs_in_accumulator else program.vector_to_accumulator("lhs", lhs)
    )
    rhs_accumulator = (
        rhs if rhs_in_accumulator else program.vector_to_accumulator("rhs", rhs)
    )
    result_accumulator = program.binary(
        "result", lhs_accumulator, rhs_accumulator, operation_descriptor_key
    )
    if result_in_accumulator:
        accumulator_unit = program.temporary("result_accumulator_unit")
        program.emits.extend(
            (
                EmitRegisterSlice(
                    source=result_accumulator,
                    result=accumulator_unit,
                    unit_count=1,
                ),
                EmitRegisterCopy(source=accumulator_unit, result=result),
            )
        )
    else:
        program.accumulator_to_vector("result", result_accumulator, result)
    return tuple(program.emits)


def emit_f32_scalar_accumulator_binary(
    lhs: ValueRef,
    rhs: ValueRef,
    result: ValueRef,
    operation_descriptor_key: str,
    *,
    temporary_prefix: str = "",
) -> tuple[ContractEmit, ...]:
    """Builds one scalar binary32 add/sub through the native accumulator."""

    def temporary(name: str) -> ValueRef:
        return ValueRef.temporary(f"{temporary_prefix}{name}")

    broadcast = _descriptor("amd.xdna.aie2p.splat.i32x16")
    return (
        _op_emit(
            broadcast,
            operands={"src": lhs},
            results={"dst": temporary("lhs_vector")},
            result_types={"dst": DescriptorResultType()},
        ),
        _op_emit(
            broadcast,
            operands={"src": rhs},
            results={"dst": temporary("rhs_vector")},
            result_types={"dst": DescriptorResultType()},
        ),
        *emit_f32_vector_accumulator_binary(
            temporary("lhs_vector"),
            temporary("rhs_vector"),
            result,
            operation_descriptor_key,
            extract_scalar_result=True,
            temporary_prefix=temporary_prefix,
        ),
    )
