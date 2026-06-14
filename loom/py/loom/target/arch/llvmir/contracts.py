# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLVMIR generic source-to-low contract fragment."""

from __future__ import annotations

from loom.dialect.buffer import ALL_BUFFER_OPS
from loom.dialect.buffer import defs as buffer
from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.index import defs as index
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scf import ALL_SCF_OPS
from loom.dialect.scf import defs as scf
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dialect.view import ALL_VIEW_OPS
from loom.dialect.view import defs as view
from loom.dsl import Op
from loom.target.arch.llvmir.descriptors import LLVMIR_GENERIC_CORE_DESCRIPTOR_SET
from loom.target.contracts import (
    AttrProject,
    ContractFragment,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    Scalar,
    SourceMemoryConstraint,
    SourceMemoryDynamicIndexSource,
    SourceMemoryOperation,
    SourceMemoryProject,
    SourceMemoryRootKind,
    TypePattern,
    ValueAliasRule,
    ValueProject,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_I1 = Scalar("i1")
_I32 = Scalar("i32")
_I64 = Scalar("i64")
_F32 = Scalar("f32")
_F64 = Scalar("f64")
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")

_VECTOR_LANE_COUNTS = (2, 4, 8, 16)
_VECTOR_SELECT_TYPES = ("i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64")

_INTEGER_PREDICATES = (
    "eq",
    "ne",
    "slt",
    "sle",
    "sgt",
    "sge",
    "ult",
    "ule",
    "ugt",
    "uge",
)

_FLOAT_PREDICATES = (
    "oeq",
    "ogt",
    "oge",
    "olt",
    "ole",
    "one",
    "ord",
    "ueq",
    "ugt",
    "uge",
    "ult",
    "ule",
    "une",
    "uno",
)

_I32_MIN = -(2**31)
_I32_MAX = (2**31) - 1

_SOURCE_MEMORY_DIAGNOSTIC = GuardDiagnostic(
    subject_role="source-memory",
    subject_name="llvmir-generic",
    constraint_key="llvmir.generic.source_memory",
)
_I64_ATTR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="attr",
    subject_name="value",
    constraint_key="llvmir.constant.i64_attr",
)
_F64_ATTR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="attr",
    subject_name="value",
    constraint_key="llvmir.constant.f64_attr",
)
_PREDICATE_DIAGNOSTIC = GuardDiagnostic(
    subject_role="attr",
    subject_name="predicate",
    constraint_key="llvmir.compare.predicate",
)


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(LLVMIR_GENERIC_CORE_DESCRIPTOR_SET, key)


def _vector_type(element: str, lane_count: int) -> TypePattern:
    return Vector(element, lanes=lane_count)


def _vector_suffix(element: str, lane_count: int) -> str:
    return f"v{lane_count}{element}"


def _typed_guards(
    fields: tuple[str, ...], type_pattern: TypePattern
) -> tuple[Guard, ...]:
    return tuple(Guard.value_type(field, type_pattern) for field in fields)


def _op_emit(
    *,
    descriptor: Descriptor,
    operands: dict[str, ValueRef] | None = None,
    results: dict[str, ValueRef] | None = None,
    immediates: dict[str, AttrProject | SourceMemoryProject | ValueProject | int]
    | None = None,
    source_memory: SourceMemoryConstraint | None = None,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={} if operands is None else operands,
        results={} if results is None else results,
        immediates={} if immediates is None else immediates,
        form=DescriptorEmitForm.OP,
        source_memory=source_memory,
    )


def _const_i32_rule(source_op: Op, result_type: TypePattern) -> DescriptorRule:
    descriptor = _descriptor("llvmir.const.i32")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64", diagnostic=_I64_ATTR_DIAGNOSTIC),
            Guard.value_type("result", result_type),
            Guard.i64_range("value", _I32_MIN, _I32_MAX),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"value": AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _const_i64_rule(source_op: Op, result_type: TypePattern) -> DescriptorRule:
    descriptor = _descriptor("llvmir.const.i64")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64", diagnostic=_I64_ATTR_DIAGNOSTIC),
            Guard.value_type("result", result_type),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"value": AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _const_float_rule(result_type: TypePattern, descriptor_key: str) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    bits_project = (
        ValueProject.f64_as_f32_bits("result")
        if result_type == _F32
        else ValueProject.f64_as_f64_bits("result")
    )
    return DescriptorRule(
        source_op=scalar_conversion.scalar_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "f64", diagnostic=_F64_ATTR_DIAGNOSTIC),
            Guard.value_type("result", result_type),
            Guard.value_exact_f64("result"),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"bits": bits_project},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _binary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_typed_guards(("lhs", "rhs", "result"), type_pattern),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _compare_rule(
    source_op: Op,
    operand_type: TypePattern,
    result_type: TypePattern,
    predicate: str,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals(
                "predicate",
                predicate,
                diagnostic=_PREDICATE_DIAGNOSTIC,
            ),
            *_typed_guards(("lhs", "rhs"), operand_type),
            Guard.value_type("result", result_type),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _select_rule(type_pattern: TypePattern, descriptor_key: str) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    vector_select = type_pattern.kind == "vector"
    condition_type = Vector("i1", lanes=type_pattern.lanes) if vector_select else _I1
    return DescriptorRule(
        source_op=vector.vector_select if vector_select else scf.scf_select,
        descriptor=descriptor,
        guards=(
            Guard.value_type("condition", condition_type),
            *_typed_guards(("true_value", "false_value", "result"), type_pattern),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "condition": ValueRef.operand("condition"),
                    "true_value": ValueRef.operand("true_value"),
                    "false_value": ValueRef.operand("false_value"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _source_memory_constraint(
    operation: SourceMemoryOperation,
    *,
    dynamic: bool,
    element_byte_count: int,
    vector_lane_count: int = 1,
) -> SourceMemoryConstraint:
    return SourceMemoryConstraint(
        operation=operation,
        root_kind=SourceMemoryRootKind.BLOCK_ARGUMENT,
        memory_spaces=("unknown", "generic", "global"),
        element_byte_count=element_byte_count,
        vector_lane_count=vector_lane_count,
        vector_lane_byte_stride=element_byte_count,
        static_byte_offset_minimum=-(2**63),
        static_byte_offset_maximum=(2**63) - 1,
        dynamic_term_count=1 if dynamic else 0,
        dynamic_index_source=(
            SourceMemoryDynamicIndexSource.VALUE
            if dynamic
            else SourceMemoryDynamicIndexSource.NONE
        ),
        dynamic_byte_stride=element_byte_count * vector_lane_count if dynamic else 0,
        diagnostic=_SOURCE_MEMORY_DIAGNOSTIC,
    )


def _memory_immediates(dynamic: bool) -> dict[str, SourceMemoryProject]:
    immediates = {"byte_offset": SourceMemoryProject.static_byte_offset()}
    if dynamic:
        immediates["byte_stride"] = SourceMemoryProject.dynamic_byte_stride()
    return immediates


def _view_load_rule(
    result_type: TypePattern,
    descriptor_key: str,
    *,
    dynamic: bool,
    element_byte_count: int,
    vector_lane_count: int = 1,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    operands = {"ptr": ValueRef.operand("view")}
    if dynamic:
        operands["index"] = ValueRef.operand("indices")
    return DescriptorRule(
        source_op=view.view_load,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("indices", 1 if dynamic else 0),
            Guard.value_type("result", result_type),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands=operands,
                results={"dst": ValueRef.result("result")},
                immediates=_memory_immediates(dynamic),
                source_memory=_source_memory_constraint(
                    SourceMemoryOperation.LOAD,
                    dynamic=dynamic,
                    element_byte_count=element_byte_count,
                    vector_lane_count=vector_lane_count,
                ),
            ),
        ),
    )


def _view_store_rule(
    value_type: TypePattern,
    descriptor_key: str,
    *,
    dynamic: bool,
    element_byte_count: int,
    vector_lane_count: int = 1,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    operands = {
        "value": ValueRef.operand("value"),
        "ptr": ValueRef.operand("view"),
    }
    if dynamic:
        operands["index"] = ValueRef.operand("indices")
    return DescriptorRule(
        source_op=view.view_store,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("indices", 1 if dynamic else 0),
            Guard.value_type("value", value_type),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands=operands,
                immediates=_memory_immediates(dynamic),
                source_memory=_source_memory_constraint(
                    SourceMemoryOperation.STORE,
                    dynamic=dynamic,
                    element_byte_count=element_byte_count,
                    vector_lane_count=vector_lane_count,
                ),
            ),
        ),
    )


def _memory_descriptor_key(operation: str, suffix: str, *, dynamic: bool) -> str:
    return (
        f"llvmir.{operation}.indexed.{suffix}"
        if dynamic
        else f"llvmir.{operation}.{suffix}"
    )


def _buffer_view_rule() -> ValueAliasRule:
    return ValueAliasRule(
        source_op=buffer.buffer_view,
        source=ValueRef.operand("buffer"),
        result=ValueRef.result("result"),
    )


def _scalar_arithmetic_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for type_pattern, suffix in ((_I32, "i32"), (_I64, "i64")):
        for source_op, stem in (
            (scalar_arithmetic.scalar_addi, "add"),
            (scalar_arithmetic.scalar_subi, "sub"),
            (scalar_arithmetic.scalar_muli, "mul"),
        ):
            rules.append(
                _binary_rule(source_op, type_pattern, f"llvmir.{stem}.{suffix}")
            )
    for type_pattern, suffix in ((_F32, "f32"), (_F64, "f64")):
        for source_op, stem in (
            (scalar_arithmetic.scalar_addf, "add"),
            (scalar_arithmetic.scalar_subf, "sub"),
            (scalar_arithmetic.scalar_mulf, "mul"),
            (scalar_arithmetic.scalar_divf, "div"),
        ):
            rules.append(
                _binary_rule(source_op, type_pattern, f"llvmir.{stem}.{suffix}")
            )
    return tuple(rules)


def _scalar_bitwise_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for type_pattern, suffix in ((_I1, "i1"), (_I32, "i32"), (_I64, "i64")):
        for source_op, stem in (
            (scalar_bitwise.scalar_andi, "and"),
            (scalar_bitwise.scalar_ori, "or"),
            (scalar_bitwise.scalar_xori, "xor"),
        ):
            rules.append(
                _binary_rule(source_op, type_pattern, f"llvmir.{stem}.{suffix}")
            )
    for type_pattern, suffix in ((_I32, "i32"), (_I64, "i64")):
        for source_op, stem in (
            (scalar_bitwise.scalar_shli, "shl"),
            (scalar_bitwise.scalar_shrui, "lshr"),
            (scalar_bitwise.scalar_shrsi, "ashr"),
        ):
            rules.append(
                _binary_rule(source_op, type_pattern, f"llvmir.{stem}.{suffix}")
            )
    return tuple(rules)


def _index_bitwise_rules() -> tuple[DescriptorRule, ...]:
    return tuple(
        _binary_rule(source_op, _INDEX, f"llvmir.{stem}.i64")
        for source_op, stem in (
            (index.index_andi, "and"),
            (index.index_ori, "or"),
            (index.index_xori, "xor"),
            (index.index_shli, "shl"),
            (index.index_shrui, "lshr"),
            (index.index_shrsi, "ashr"),
        )
    )


def _vector_arithmetic_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for element, source_ops in (
        (
            "i32",
            (
                (vector.vector_addi, "add"),
                (vector.vector_subi, "sub"),
                (vector.vector_muli, "mul"),
            ),
        ),
        (
            "f32",
            (
                (vector.vector_addf, "add"),
                (vector.vector_subf, "sub"),
                (vector.vector_mulf, "mul"),
            ),
        ),
    ):
        for lane_count in _VECTOR_LANE_COUNTS:
            type_pattern = _vector_type(element, lane_count)
            suffix = _vector_suffix(element, lane_count)
            for source_op, stem in source_ops:
                rules.append(
                    _binary_rule(source_op, type_pattern, f"llvmir.{stem}.{suffix}")
                )
    return tuple(rules)


def _vector_bitwise_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for element, source_ops in (
        (
            "i1",
            (
                (vector.vector_andi, "and"),
                (vector.vector_ori, "or"),
                (vector.vector_xori, "xor"),
            ),
        ),
        (
            "i32",
            (
                (vector.vector_andi, "and"),
                (vector.vector_ori, "or"),
                (vector.vector_xori, "xor"),
                (vector.vector_shli, "shl"),
                (vector.vector_shrui, "lshr"),
                (vector.vector_shrsi, "ashr"),
            ),
        ),
    ):
        for lane_count in _VECTOR_LANE_COUNTS:
            type_pattern = _vector_type(element, lane_count)
            suffix = _vector_suffix(element, lane_count)
            for source_op, stem in source_ops:
                rules.append(
                    _binary_rule(source_op, type_pattern, f"llvmir.{stem}.{suffix}")
                )
    return tuple(rules)


def _compare_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for type_pattern, suffix, result_type, source_op, predicates in (
        (_I32, "i32", _I1, scalar_comparison.scalar_cmpi, _INTEGER_PREDICATES),
        (_I64, "i64", _I1, scalar_comparison.scalar_cmpi, _INTEGER_PREDICATES),
        (_F32, "f32", _I1, scalar_comparison.scalar_cmpf, _FLOAT_PREDICATES),
        (_F64, "f64", _I1, scalar_comparison.scalar_cmpf, _FLOAT_PREDICATES),
    ):
        rules.extend(
            (
                _compare_rule(
                    source_op,
                    type_pattern,
                    result_type,
                    predicate,
                    f"llvmir.cmp.{predicate}.{suffix}",
                )
            )
            for predicate in predicates
        )
    for lane_count in _VECTOR_LANE_COUNTS:
        mask_type = _vector_type("i1", lane_count)
        for element, source_op, predicates in (
            ("i32", vector.vector_cmpi, _INTEGER_PREDICATES),
            ("f32", vector.vector_cmpf, _FLOAT_PREDICATES),
        ):
            value_type = _vector_type(element, lane_count)
            suffix = _vector_suffix(element, lane_count)
            rules.extend(
                (
                    _compare_rule(
                        source_op,
                        value_type,
                        mask_type,
                        predicate,
                        f"llvmir.cmp.{predicate}.{suffix}",
                    )
                )
                for predicate in predicates
            )
    return tuple(rules)


def _select_rules() -> tuple[DescriptorRule, ...]:
    return tuple(
        _select_rule(type_pattern, f"llvmir.select.{suffix}")
        for type_pattern, suffix in (
            (_I32, "i32"),
            (_I64, "i64"),
            (_F32, "f32"),
            (_F64, "f64"),
        )
    ) + tuple(
        _select_rule(
            _vector_type(element, lane_count),
            f"llvmir.select.{_vector_suffix(element, lane_count)}",
        )
        for element in _VECTOR_SELECT_TYPES
        for lane_count in _VECTOR_LANE_COUNTS
    )


def _memory_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for value_type, suffix, element_byte_count, vector_lane_count in (
        (_I32, "i32", 4, 1),
        (_I64, "i64", 8, 1),
        (_F32, "f32", 4, 1),
        (_F64, "f64", 8, 1),
        (_vector_type("i32", 4), "v4i32", 4, 4),
        (_vector_type("f32", 4), "v4f32", 4, 4),
    ):
        for dynamic in (False, True):
            rules.append(
                _view_load_rule(
                    value_type,
                    _memory_descriptor_key("load", suffix, dynamic=dynamic),
                    dynamic=dynamic,
                    element_byte_count=element_byte_count,
                    vector_lane_count=vector_lane_count,
                )
            )
            rules.append(
                _view_store_rule(
                    value_type,
                    _memory_descriptor_key("store", suffix, dynamic=dynamic),
                    dynamic=dynamic,
                    element_byte_count=element_byte_count,
                    vector_lane_count=vector_lane_count,
                )
            )
    return tuple(rules)


LLVMIR_GENERIC_CORE_CONTRACT_DIALECT_OPS = {
    "buffer": ALL_BUFFER_OPS,
    "index": ALL_INDEX_OPS,
    "scalar": ALL_SCALAR_OPS,
    "scf": ALL_SCF_OPS,
    "vector": ALL_VECTOR_OPS,
    "view": ALL_VIEW_OPS,
}

LLVMIR_GENERIC_CORE_CONTRACT_FRAGMENT = ContractFragment(
    name="llvmir.generic.core",
    descriptor_set=LLVMIR_GENERIC_CORE_DESCRIPTOR_SET,
    public_header="loom/target/arch/llvmir/contracts/generic_core.h",
    cases=(
        _const_i32_rule(scalar_conversion.scalar_constant, _I32),
        _const_i32_rule(index.index_constant, _INDEX),
        _const_i64_rule(scalar_conversion.scalar_constant, _I64),
        _const_i64_rule(index.index_constant, _OFFSET),
        _const_float_rule(_F32, "llvmir.const.f32"),
        _const_float_rule(_F64, "llvmir.const.f64"),
        ValueAliasRule(
            source_op=index.index_assume,
            source=ValueRef.operand("values"),
            result=ValueRef.result("results"),
            guards=(Guard.operand_segment_count("values", 1),),
        ),
        _buffer_view_rule(),
        *_scalar_arithmetic_rules(),
        *_scalar_bitwise_rules(),
        _binary_rule(index.index_add, _INDEX, "llvmir.add.i64"),
        _binary_rule(index.index_sub, _INDEX, "llvmir.sub.i64"),
        _binary_rule(index.index_mul, _INDEX, "llvmir.mul.i64"),
        *_index_bitwise_rules(),
        _binary_rule(index.index_add, _OFFSET, "llvmir.add.i64"),
        _binary_rule(index.index_sub, _OFFSET, "llvmir.sub.i64"),
        *_vector_arithmetic_rules(),
        *_vector_bitwise_rules(),
        *_compare_rules(),
        *_select_rules(),
        *_memory_rules(),
    ),
)
