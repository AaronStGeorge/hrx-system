# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""x86 lowering-only scalar read-memory contract fragment."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.view import ALL_VIEW_OPS
from loom.dialect.view import defs as view
from loom.dsl import Op
from loom.target.arch.x86.contracts.memory import x86_fused_load_rules
from loom.target.arch.x86.descriptors import X86_SCALAR_DESCRIPTOR_SET
from loom.target.arch.x86.scalar_families import (
    X86_SCALAR_MEMORY_EXTENSION_FAMILIES,
    X86_SCALAR_READ_MEMORY_FAMILIES,
    ScalarMemoryExtensionFamily,
)
from loom.target.contracts import (
    ContractCase,
    ContractFragment,
    DescriptorRule,
    Guard,
    GuardDiagnostic,
    Scalar,
    SourceNode,
    TypePattern,
    ValueRef,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_I32 = Scalar("i32")
_I64 = Scalar("i64")

# Two-node graphs score above the ordinary one-node load and arithmetic rules.
# Immediate arithmetic uses priority three in the scalar legality fragment and
# therefore remains preferable when both realizations are available.
_READ_MEMORY_PRIORITY = 1

_SOURCE_MEMORY_DIAGNOSTIC = GuardDiagnostic(
    subject_role="source-memory",
    subject_name="x86-scalar",
    constraint_key="x86.scalar.source_memory",
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(X86_SCALAR_DESCRIPTOR_SET, key)


def _typed_guards(
    fields: Iterable[str],
    type_pattern: TypePattern,
) -> tuple[Guard, ...]:
    return tuple(Guard.value_type(field, type_pattern) for field in fields)


def _read_memory_rules(
    source_op: Op,
    type_pattern: TypePattern,
    operation: str,
    register_width: int,
    *,
    loaded_operands: Sequence[str] = ("lhs", "rhs"),
) -> Iterable[DescriptorRule]:
    for loaded_operand in loaded_operands:
        accumulator_operand = "rhs" if loaded_operand == "lhs" else "lhs"
        consumer = SourceNode.exclusive_user(
            "consumer",
            source_op=source_op,
            parent_result=ValueRef.result("result"),
            node_operand=ValueRef.operand(loaded_operand),
            guards=_typed_guards(("lhs", "rhs", "result"), type_pattern),
        )
        yield from x86_fused_load_rules(
            _descriptor,
            source_op=view.view_load,
            source_type=type_pattern,
            source_nodes=(consumer,),
            result=ValueRef.result("result", source_node="consumer"),
            element_byte_count=register_width // 8,
            lane_count=1,
            descriptor_key_prefix=f"x86.scalar.{operation}",
            register_suffix=f"gpr{register_width}",
            instruction_operands={
                "lhs": ValueRef.operand(accumulator_operand, source_node="consumer")
            },
            diagnostic=_SOURCE_MEMORY_DIAGNOSTIC,
            report_key=(
                f"native_memory_integer_{operation}_i{register_width}_{loaded_operand}"
            ),
            priority=_READ_MEMORY_PRIORITY,
        )


def _memory_extension_rules(
    family: ScalarMemoryExtensionFamily,
) -> Iterable[DescriptorRule]:
    source_type = Scalar(f"i{family.source_bit_count}")
    result_type = Scalar(
        tuple(f"i{bit_count}" for bit_count in family.result_bit_counts)
    )
    operation = "extsi" if family.signed else "extui"
    source_prefix = "i" if family.signed else "u"
    extension = SourceNode.exclusive_user(
        "extension",
        source_op=(
            scalar_conversion.scalar_extsi
            if family.signed
            else scalar_conversion.scalar_extui
        ),
        parent_result=ValueRef.result("result"),
        node_operand=ValueRef.operand("input"),
        guards=(
            Guard.value_type("input", source_type),
            Guard.value_type("result", result_type),
        ),
    )
    yield from x86_fused_load_rules(
        _descriptor,
        source_op=view.view_load,
        source_type=source_type,
        source_nodes=(extension,),
        result=ValueRef.result("result", source_node="extension"),
        element_byte_count=family.source_bit_count // 8,
        lane_count=1,
        descriptor_key_prefix=f"x86.scalar.{family.mnemonic}",
        register_suffix=(
            f"{source_prefix}{family.source_bit_count}.gpr{family.carrier_bit_count}"
        ),
        diagnostic=_SOURCE_MEMORY_DIAGNOSTIC,
        report_key=(
            f"native_memory_extension_{operation}_i{family.source_bit_count}_"
            + "_".join(f"i{bit_count}" for bit_count in family.result_bit_counts)
        ),
        priority=_READ_MEMORY_PRIORITY,
    )


def _cases() -> tuple[ContractCase, ...]:
    source_operations = {
        "addi": scalar_arithmetic.scalar_addi,
        "subi": scalar_arithmetic.scalar_subi,
        "muli": scalar_arithmetic.scalar_muli,
        "andi": scalar_bitwise.scalar_andi,
        "ori": scalar_bitwise.scalar_ori,
        "xori": scalar_bitwise.scalar_xori,
    }
    read_memory_rules = tuple(
        rule
        for type_pattern, register_width in ((_I32, 32), (_I64, 64))
        for family in X86_SCALAR_READ_MEMORY_FAMILIES
        for rule in _read_memory_rules(
            source_operations[family.source_operation],
            type_pattern,
            family.descriptor_operation,
            register_width,
            loaded_operands=(("lhs", "rhs") if family.commutative else ("rhs",)),
        )
    )
    return (
        *read_memory_rules,
        *(
            rule
            for family in X86_SCALAR_MEMORY_EXTENSION_FAMILIES
            for rule in _memory_extension_rules(family)
        ),
    )


X86_SCALAR_READ_MEMORY_CONTRACT_DIALECT_OPS = {
    "scalar": ALL_SCALAR_OPS,
    "view": ALL_VIEW_OPS,
}

X86_SCALAR_READ_MEMORY_CONTRACT_FRAGMENT = ContractFragment(
    name="x86.scalar.read_memory",
    descriptor_set=X86_SCALAR_DESCRIPTOR_SET,
    public_header="loom/target/arch/x86/contracts/scalar_read_memory.h",
    cases=_cases(),
    # These are optional lowering realizations. Each source operation has an
    # independent legality contract, so exposing their overlapping graphs to
    # speculative legalization would add no supported source program.
    target_contract_query=False,
)
