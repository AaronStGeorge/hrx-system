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
from loom.dialect.kernel import ALL_KERNEL_OPS
from loom.dialect.kernel import defs as kernel
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scalar import math as scalar_math
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
    ResultTypeBinding,
    Scalar,
    SourceMemoryByteOffsetMaterializer,
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
_F16 = Scalar("f16")
_BF16 = Scalar("bf16")
_F32 = Scalar("f32")
_F64 = Scalar("f64")
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")

_VECTOR_LANE_COUNTS = (2, 4, 8, 16)
_VECTOR_SELECT_TYPES = ("i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64")
_STRUCTURAL_VECTOR_TYPES = (
    "i1",
    "i8",
    "i16",
    "i32",
    "i64",
    "f16",
    "bf16",
    "f32",
    "f64",
)
_MEMORY_VALUE_TYPES = (
    ("i8", 1),
    ("i16", 2),
    ("i32", 4),
    ("i64", 8),
    ("f16", 2),
    ("bf16", 2),
    ("f32", 4),
    ("f64", 8),
)

_KERNEL_DIMENSIONS = ("x", "y", "z")

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

_SCALAR_CAST_SPECS = (
    (scalar_conversion.scalar_trunci, "trunc", "i32", "i8"),
    (scalar_conversion.scalar_trunci, "trunc", "i32", "i16"),
    (scalar_conversion.scalar_trunci, "trunc", "i64", "i32"),
    (scalar_conversion.scalar_extsi, "sext", "i8", "i32"),
    (scalar_conversion.scalar_extsi, "sext", "i16", "i32"),
    (scalar_conversion.scalar_extsi, "sext", "i32", "i64"),
    (scalar_conversion.scalar_extui, "zext", "i8", "i32"),
    (scalar_conversion.scalar_extui, "zext", "i16", "i32"),
    (scalar_conversion.scalar_extui, "zext", "i32", "i64"),
    (scalar_conversion.scalar_sitofp, "sitofp", "i8", "f32"),
    (scalar_conversion.scalar_sitofp, "sitofp", "i32", "f32"),
    (scalar_conversion.scalar_sitofp, "sitofp", "i64", "f64"),
    (scalar_conversion.scalar_uitofp, "uitofp", "i8", "f32"),
    (scalar_conversion.scalar_uitofp, "uitofp", "i32", "f32"),
    (scalar_conversion.scalar_uitofp, "uitofp", "i64", "f64"),
    (scalar_conversion.scalar_fptosi, "fptosi", "f32", "i32"),
    (scalar_conversion.scalar_fptosi, "fptosi", "f64", "i64"),
    (scalar_conversion.scalar_fptoui, "fptoui", "f32", "i32"),
    (scalar_conversion.scalar_fptoui, "fptoui", "f64", "i64"),
    (scalar_conversion.scalar_fptrunc, "fptrunc", "f32", "f16"),
    (scalar_conversion.scalar_fptrunc, "fptrunc", "f32", "bf16"),
    (scalar_conversion.scalar_fptrunc, "fptrunc", "f64", "f32"),
    (scalar_conversion.scalar_extf, "fpext", "f16", "f32"),
    (scalar_conversion.scalar_extf, "fpext", "bf16", "f32"),
    (scalar_conversion.scalar_extf, "fpext", "f32", "f64"),
    (scalar_conversion.scalar_bitcast, "bitcast", "i16", "f16"),
    (scalar_conversion.scalar_bitcast, "bitcast", "i16", "bf16"),
    (scalar_conversion.scalar_bitcast, "bitcast", "f16", "i16"),
    (scalar_conversion.scalar_bitcast, "bitcast", "bf16", "i16"),
    (scalar_conversion.scalar_bitcast, "bitcast", "f16", "bf16"),
    (scalar_conversion.scalar_bitcast, "bitcast", "bf16", "f16"),
    (scalar_conversion.scalar_bitcast, "bitcast", "i32", "f32"),
    (scalar_conversion.scalar_bitcast, "bitcast", "f32", "i32"),
    (scalar_conversion.scalar_bitcast, "bitcast", "i64", "f64"),
    (scalar_conversion.scalar_bitcast, "bitcast", "f64", "i64"),
)

_VECTOR_CAST_SPECS = (
    (vector.vector_extsi, "sext", "i32", "i64"),
    (vector.vector_extui, "zext", "i32", "i64"),
    (vector.vector_trunci, "trunc", "i64", "i32"),
    (vector.vector_sitofp, "sitofp", "i32", "f32"),
    (vector.vector_uitofp, "uitofp", "i32", "f32"),
    (vector.vector_fptosi, "fptosi", "f32", "i32"),
    (vector.vector_fptoui, "fptoui", "f32", "i32"),
    (vector.vector_fptrunc, "fptrunc", "f32", "f16"),
    (vector.vector_fptrunc, "fptrunc", "f32", "bf16"),
    (vector.vector_extf, "fpext", "f16", "f32"),
    (vector.vector_extf, "fpext", "bf16", "f32"),
    (vector.vector_bitcast, "bitcast", "i32", "f32"),
    (vector.vector_bitcast, "bitcast", "f32", "i32"),
    (vector.vector_bitcast, "bitcast", "i64", "f64"),
    (vector.vector_bitcast, "bitcast", "f64", "i64"),
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


def _source_memory_byte_offset_materializer() -> SourceMemoryByteOffsetMaterializer:
    return SourceMemoryByteOffsetMaterializer(
        const_i64=_descriptor("llvmir.const.i64"),
        add_i64=_descriptor("llvmir.add.i64"),
        mul_i64=_descriptor("llvmir.mul.i64"),
        shl_i64=_descriptor("llvmir.shl.i64"),
    )


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
    result_types: dict[str, ResultTypeBinding] | None = None,
    immediates: dict[str, AttrProject | SourceMemoryProject | ValueProject | int]
    | None = None,
    source_memory: SourceMemoryConstraint | None = None,
    source_memory_byte_offset_materializer: (
        SourceMemoryByteOffsetMaterializer | None
    ) = None,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={} if operands is None else operands,
        results={} if results is None else results,
        result_types=result_types,
        immediates={} if immediates is None else immediates,
        form=DescriptorEmitForm.OP,
        source_memory=source_memory,
        source_memory_byte_offset_materializer=source_memory_byte_offset_materializer,
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


def _const_float_bits_project(element: str) -> ValueProject:
    if element == "f16":
        return ValueProject.f64_as_f16_bits("result")
    if element == "bf16":
        return ValueProject.f64_as_bf16_bits("result")
    if element == "f32":
        return ValueProject.f64_as_f32_bits("result")
    if element == "f64":
        return ValueProject.f64_as_f64_bits("result")
    raise ValueError(f"unsupported LLVMIR float constant type '{element}'")


def _const_float_rule(
    source_op: Op, result_type: TypePattern, descriptor_key: str
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    bits_project = _const_float_bits_project(result_type.element)
    return DescriptorRule(
        source_op=source_op,
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


def _vector_const_i_rule(
    element: str,
    lane_count: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> DescriptorRule:
    descriptor = _descriptor(f"llvmir.const.{_vector_suffix(element, lane_count)}")
    guards = [
        Guard.attr_kind("value", "i64", diagnostic=_I64_ATTR_DIAGNOSTIC),
        Guard.value_type("result", _vector_type(element, lane_count)),
    ]
    if minimum is not None and maximum is not None:
        guards.append(Guard.i64_range("value", minimum, maximum))
    return DescriptorRule(
        source_op=vector.vector_constant,
        descriptor=descriptor,
        guards=tuple(guards),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"value": AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _vector_const_float_rule(element: str, lane_count: int) -> DescriptorRule:
    result_type = _vector_type(element, lane_count)
    descriptor = _descriptor(f"llvmir.const.{_vector_suffix(element, lane_count)}")
    bits_project = _const_float_bits_project(element)
    return DescriptorRule(
        source_op=vector.vector_constant,
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


def _unary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_typed_guards(("input", "result"), type_pattern),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"input": ValueRef.operand("input")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _ternary_rule(
    source_op: Op,
    type_pattern: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_typed_guards(("a", "b", "c", "result"), type_pattern),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "a": ValueRef.operand("a"),
                    "b": ValueRef.operand("b"),
                    "c": ValueRef.operand("c"),
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


def _cast_rule(
    source_op: Op,
    source_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.value_type("input", source_type),
            Guard.value_type("result", result_type),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"value": ValueRef.operand("input")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _index_cast_rule(
    source_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    return _cast_rule(index.index_cast, source_type, result_type, descriptor_key)


def _index_cast_alias_rule(
    source_type: TypePattern,
    result_type: TypePattern,
) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=index.index_cast,
        source=ValueRef.operand("input"),
        result=ValueRef.result("result"),
        guards=(
            Guard.value_type("input", source_type),
            Guard.value_type("result", result_type),
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
    dynamic_term_count: int,
    element_byte_count: int,
    vector_lane_count: int = 1,
) -> SourceMemoryConstraint:
    if dynamic_term_count == 0:
        dynamic_index_source = SourceMemoryDynamicIndexSource.NONE
        dynamic_byte_stride: int | None = 0
    else:
        dynamic_index_source = SourceMemoryDynamicIndexSource.VALUE
        dynamic_byte_stride = None
    return SourceMemoryConstraint(
        operation=operation,
        root_kind=SourceMemoryRootKind.ANY,
        memory_spaces=(
            "unknown",
            "generic",
            "global",
            "workgroup",
            "private",
            "constant",
            "descriptor",
        ),
        element_byte_count=element_byte_count,
        vector_lane_count=vector_lane_count,
        vector_lane_byte_stride=element_byte_count,
        static_byte_offset_minimum=-(2**63),
        static_byte_offset_maximum=(2**63) - 1,
        dynamic_term_count=dynamic_term_count,
        dynamic_index_source=dynamic_index_source,
        dynamic_byte_stride=dynamic_byte_stride,
        diagnostic=_SOURCE_MEMORY_DIAGNOSTIC,
    )


def _memory_immediates(
    *,
    dynamic_term_count: int,
    materialize_byte_offset: bool,
) -> dict[str, SourceMemoryProject | int]:
    immediates: dict[str, SourceMemoryProject | int] = {
        "byte_offset": SourceMemoryProject.static_byte_offset()
    }
    if dynamic_term_count != 0:
        immediates["byte_stride"] = (
            1 if materialize_byte_offset else SourceMemoryProject.dynamic_byte_stride()
        )
    return immediates


def _view_load_rule(
    source_op: Op,
    view_field: str,
    result_type: TypePattern,
    descriptor_key: str,
    *,
    source_dynamic_count: int,
    dynamic_term_count: int,
    element_byte_count: int,
    vector_lane_count: int = 1,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    materialize_byte_offset = dynamic_term_count > 1
    operands = {"ptr": ValueRef.operand(view_field)}
    if dynamic_term_count != 0:
        if materialize_byte_offset:
            operands["index"] = ValueRef.source_memory_dynamic_byte_offset()
        elif source_dynamic_count:
            operands["index"] = ValueRef.operand("indices")
        else:
            operands["index"] = ValueRef.source_memory_dynamic_term()
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("indices", source_dynamic_count),
            Guard.value_type("result", result_type),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands=operands,
                results={"dst": ValueRef.result("result")},
                immediates=_memory_immediates(
                    dynamic_term_count=dynamic_term_count,
                    materialize_byte_offset=materialize_byte_offset,
                ),
                source_memory=_source_memory_constraint(
                    SourceMemoryOperation.LOAD,
                    dynamic_term_count=dynamic_term_count,
                    element_byte_count=element_byte_count,
                    vector_lane_count=vector_lane_count,
                ),
                source_memory_byte_offset_materializer=(
                    _source_memory_byte_offset_materializer()
                    if materialize_byte_offset
                    else None
                ),
            ),
        ),
    )


def _view_store_rule(
    source_op: Op,
    view_field: str,
    value_type: TypePattern,
    descriptor_key: str,
    *,
    source_dynamic_count: int,
    dynamic_term_count: int,
    element_byte_count: int,
    vector_lane_count: int = 1,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    materialize_byte_offset = dynamic_term_count > 1
    operands = {
        "value": ValueRef.operand("value"),
        "ptr": ValueRef.operand(view_field),
    }
    if dynamic_term_count != 0:
        if materialize_byte_offset:
            operands["index"] = ValueRef.source_memory_dynamic_byte_offset()
        elif source_dynamic_count:
            operands["index"] = ValueRef.operand("indices")
        else:
            operands["index"] = ValueRef.source_memory_dynamic_term()
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("indices", source_dynamic_count),
            Guard.value_type("value", value_type),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands=operands,
                immediates=_memory_immediates(
                    dynamic_term_count=dynamic_term_count,
                    materialize_byte_offset=materialize_byte_offset,
                ),
                source_memory=_source_memory_constraint(
                    SourceMemoryOperation.STORE,
                    dynamic_term_count=dynamic_term_count,
                    element_byte_count=element_byte_count,
                    vector_lane_count=vector_lane_count,
                ),
                source_memory_byte_offset_materializer=(
                    _source_memory_byte_offset_materializer()
                    if materialize_byte_offset
                    else None
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


def _buffer_alloca_rule(memory_space: str) -> DescriptorRule:
    descriptor = _descriptor(f"llvmir.alloca.{memory_space}.i8")
    return DescriptorRule(
        source_op=buffer.buffer_alloca,
        descriptor=descriptor,
        guards=(
            Guard.value_type("byte_length", _OFFSET),
            Guard.enum_attr_equals("memory_space", memory_space),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"byte_length": ValueRef.operand("byte_length")},
                results={"ptr": ValueRef.result("result")},
                immediates={
                    "base_alignment": AttrProject.direct("base_alignment"),
                },
            ),
        ),
    )


def _scalar_arithmetic_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for type_pattern, suffix in ((_I32, "i32"), (_I64, "i64")):
        for source_op, stem in (
            (scalar_arithmetic.scalar_addi, "add"),
            (scalar_arithmetic.scalar_subi, "sub"),
            (scalar_arithmetic.scalar_muli, "mul"),
            (scalar_arithmetic.scalar_divui, "udiv"),
            (scalar_arithmetic.scalar_divsi, "sdiv"),
            (scalar_arithmetic.scalar_remui, "urem"),
            (scalar_arithmetic.scalar_remsi, "srem"),
        ):
            rules.append(
                _binary_rule(source_op, type_pattern, f"llvmir.{stem}.{suffix}")
            )
        for source_op, predicate in (
            (scalar_arithmetic.scalar_minsi, "slt"),
            (scalar_arithmetic.scalar_maxsi, "sgt"),
            (scalar_arithmetic.scalar_minui, "ult"),
            (scalar_arithmetic.scalar_maxui, "ugt"),
        ):
            rules.append(
                _scalar_minmax_rule(source_op, type_pattern, suffix, predicate)
            )
    for type_pattern, suffix in ((_F32, "f32"), (_F64, "f64")):
        for source_op, stem in (
            (scalar_arithmetic.scalar_negf, "neg"),
            (scalar_arithmetic.scalar_absf, "abs"),
        ):
            rules.append(
                _unary_rule(source_op, type_pattern, f"llvmir.{stem}.{suffix}")
            )
        for source_op, stem in (
            (scalar_arithmetic.scalar_addf, "add"),
            (scalar_arithmetic.scalar_subf, "sub"),
            (scalar_arithmetic.scalar_mulf, "mul"),
            (scalar_arithmetic.scalar_divf, "div"),
            (scalar_arithmetic.scalar_minnumf, "minnum"),
            (scalar_arithmetic.scalar_maxnumf, "maxnum"),
        ):
            rules.append(
                _binary_rule(source_op, type_pattern, f"llvmir.{stem}.{suffix}")
            )
        rules.append(
            _ternary_rule(
                scalar_math.scalar_fmaf,
                type_pattern,
                f"llvmir.fma.{suffix}",
            )
        )
    return tuple(rules)


def _scalar_minmax_rule(
    source_op: Op, type_pattern: TypePattern, suffix: str, predicate: str
) -> DescriptorRule:
    compare_descriptor = _descriptor(f"llvmir.cmp.{predicate}.{suffix}")
    select_descriptor = _descriptor(f"llvmir.select.{suffix}")
    return DescriptorRule(
        source_op=source_op,
        descriptor=compare_descriptor,
        guards=_typed_guards(("lhs", "rhs", "result"), type_pattern),
        emit=(
            _op_emit(
                descriptor=compare_descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.temporary("take_lhs")},
                result_types={"dst": _I1},
            ),
            _op_emit(
                descriptor=select_descriptor,
                operands={
                    "condition": ValueRef.temporary("take_lhs"),
                    "true_value": ValueRef.operand("lhs"),
                    "false_value": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


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


def _index_madd_rule() -> DescriptorRule:
    multiply = _descriptor("llvmir.mul.i64")
    add = _descriptor("llvmir.add.i64")
    return DescriptorRule(
        source_op=index.index_madd,
        descriptor=add,
        guards=_typed_guards(("a", "b", "c", "result"), _INDEX),
        emit=(
            _op_emit(
                descriptor=multiply,
                operands={
                    "lhs": ValueRef.operand("a"),
                    "rhs": ValueRef.operand("b"),
                },
                results={"dst": ValueRef.temporary("product")},
                result_types={"dst": _INDEX},
            ),
            _op_emit(
                descriptor=add,
                operands={
                    "lhs": ValueRef.temporary("product"),
                    "rhs": ValueRef.operand("c"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


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


def _index_minmax_rule(source_op: Op, predicate: str) -> DescriptorRule:
    compare_descriptor = _descriptor(f"llvmir.cmp.{predicate}.i64")
    select_descriptor = _descriptor("llvmir.select.i64")
    return DescriptorRule(
        source_op=source_op,
        descriptor=compare_descriptor,
        guards=_typed_guards(("lhs", "rhs", "result"), _INDEX),
        emit=(
            _op_emit(
                descriptor=compare_descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.temporary("take_lhs")},
                result_types={"dst": _I1},
            ),
            _op_emit(
                descriptor=select_descriptor,
                operands={
                    "condition": ValueRef.temporary("take_lhs"),
                    "true_value": ValueRef.operand("lhs"),
                    "false_value": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _index_minmax_rules() -> tuple[DescriptorRule, ...]:
    return (
        _index_minmax_rule(index.index_min, "slt"),
        _index_minmax_rule(index.index_max, "sgt"),
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
                (vector.vector_minnumf, "minnum"),
                (vector.vector_maxnumf, "maxnum"),
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
        vector_type = _vector_type(element, 1)
        for source_op, stem in source_ops:
            rules.append(
                _binary_rule(source_op, vector_type, f"llvmir.{stem}.{element}")
            )
    for lane_count in _VECTOR_LANE_COUNTS:
        type_pattern = _vector_type("f32", lane_count)
        suffix = _vector_suffix("f32", lane_count)
        for source_op, stem in (
            (vector.vector_negf, "neg"),
            (vector.vector_absf, "abs"),
        ):
            rules.append(
                _unary_rule(source_op, type_pattern, f"llvmir.{stem}.{suffix}")
            )
    for source_op, stem in (
        (vector.vector_negf, "neg"),
        (vector.vector_absf, "abs"),
    ):
        rules.append(
            _unary_rule(source_op, _vector_type("f32", 1), f"llvmir.{stem}.f32")
        )
    for lane_count in _VECTOR_LANE_COUNTS:
        type_pattern = _vector_type("f32", lane_count)
        suffix = _vector_suffix("f32", lane_count)
        rules.append(
            _ternary_rule(vector.vector_fmaf, type_pattern, f"llvmir.fma.{suffix}")
        )
    rules.append(
        _ternary_rule(
            vector.vector_fmaf,
            _vector_type("f32", 1),
            "llvmir.fma.f32",
        )
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
        vector_type = _vector_type(element, 1)
        for source_op, stem in source_ops:
            rules.append(
                _binary_rule(source_op, vector_type, f"llvmir.{stem}.{element}")
            )
    return tuple(rules)


def _compare_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for type_pattern, suffix, result_type, source_op, predicates in (
        (_INDEX, "i64", _I1, index.index_cmp, _INTEGER_PREDICATES),
        (_OFFSET, "i64", _I1, index.index_cmp, _INTEGER_PREDICATES),
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
    for element, source_op, predicates in (
        ("i32", vector.vector_cmpi, _INTEGER_PREDICATES),
        ("f32", vector.vector_cmpf, _FLOAT_PREDICATES),
    ):
        value_type = _vector_type(element, 1)
        mask_type = _vector_type("i1", 1)
        rules.extend(
            (
                _compare_rule(
                    source_op,
                    value_type,
                    mask_type,
                    predicate,
                    f"llvmir.cmp.{predicate}.{element}",
                )
            )
            for predicate in predicates
        )
    return tuple(rules)


def _cast_rules() -> tuple[DescriptorRule, ...]:
    return (
        tuple(
            _cast_rule(
                source_op,
                Scalar(source_type),
                Scalar(result_type),
                f"llvmir.{stem}.{source_type}.{result_type}",
            )
            for source_op, stem, source_type, result_type in _SCALAR_CAST_SPECS
        )
        + tuple(
            _cast_rule(
                source_op,
                _vector_type(source_type, lane_count),
                _vector_type(result_type, lane_count),
                "llvmir."
                f"{stem}."
                f"{_vector_suffix(source_type, lane_count)}."
                f"{_vector_suffix(result_type, lane_count)}",
            )
            for source_op, stem, source_type, result_type in _VECTOR_CAST_SPECS
            for lane_count in _VECTOR_LANE_COUNTS
        )
        + tuple(
            _cast_rule(
                source_op,
                _vector_type(source_type, 1),
                _vector_type(result_type, 1),
                f"llvmir.{stem}.{source_type}.{result_type}",
            )
            for source_op, stem, source_type, result_type in _VECTOR_CAST_SPECS
        )
    )


def _index_cast_rules() -> tuple[DescriptorRule | ValueAliasRule, ...]:
    return (
        _index_cast_rule(_I32, _INDEX, "llvmir.sext.i32.i64"),
        _index_cast_rule(_I32, _OFFSET, "llvmir.zext.i32.i64"),
        _index_cast_rule(_INDEX, _I32, "llvmir.trunc.i64.i32"),
        _index_cast_rule(_OFFSET, _I32, "llvmir.trunc.i64.i32"),
        _index_cast_alias_rule(_I64, _INDEX),
        _index_cast_alias_rule(_I64, _OFFSET),
        _index_cast_alias_rule(_INDEX, _I64),
        _index_cast_alias_rule(_OFFSET, _I64),
        _index_cast_alias_rule(_INDEX, _OFFSET),
        _index_cast_alias_rule(_OFFSET, _INDEX),
    )


def _select_rules() -> tuple[DescriptorRule, ...]:
    return (
        tuple(
            _select_rule(type_pattern, f"llvmir.select.{suffix}")
            for type_pattern, suffix in (
                (_INDEX, "i64"),
                (_OFFSET, "i64"),
                (_I32, "i32"),
                (_I64, "i64"),
                (_F32, "f32"),
                (_F64, "f64"),
            )
        )
        + tuple(
            _select_rule(
                _vector_type(element, lane_count),
                f"llvmir.select.{_vector_suffix(element, lane_count)}",
            )
            for element in _VECTOR_SELECT_TYPES
            for lane_count in _VECTOR_LANE_COUNTS
        )
        + tuple(
            _select_rule(_vector_type(element, 1), f"llvmir.select.{element}")
            for element in _VECTOR_SELECT_TYPES
            if element in ("i32", "i64", "f32", "f64")
        )
    )


def _clampf_number_rule(
    source_op: Op,
    type_pattern: TypePattern,
    maxnum_descriptor_key: str,
    minnum_descriptor_key: str,
) -> DescriptorRule:
    maxnum_descriptor = _descriptor(maxnum_descriptor_key)
    minnum_descriptor = _descriptor(minnum_descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=maxnum_descriptor,
        guards=(
            Guard.enum_attr_equals("mode", "number"),
            *_typed_guards(("value", "lower", "upper", "result"), type_pattern),
        ),
        emit=(
            _op_emit(
                descriptor=maxnum_descriptor,
                operands={
                    "lhs": ValueRef.operand("value"),
                    "rhs": ValueRef.operand("lower"),
                },
                results={"dst": ValueRef.temporary("lower_clamped")},
                result_types={"dst": type_pattern},
            ),
            _op_emit(
                descriptor=minnum_descriptor,
                operands={
                    "lhs": ValueRef.temporary("lower_clamped"),
                    "rhs": ValueRef.operand("upper"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _clampf_ordered_rule(
    source_op: Op,
    type_pattern: TypePattern,
    mask_pattern: TypePattern,
    lower_compare_descriptor_key: str,
    upper_compare_descriptor_key: str,
    select_descriptor_key: str,
) -> DescriptorRule:
    lower_compare_descriptor = _descriptor(lower_compare_descriptor_key)
    upper_compare_descriptor = _descriptor(upper_compare_descriptor_key)
    select_descriptor = _descriptor(select_descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=lower_compare_descriptor,
        guards=(
            Guard.enum_attr_equals("mode", "ordered"),
            *_typed_guards(("value", "lower", "upper", "result"), type_pattern),
        ),
        emit=(
            _op_emit(
                descriptor=lower_compare_descriptor,
                operands={
                    "lhs": ValueRef.operand("value"),
                    "rhs": ValueRef.operand("lower"),
                },
                results={"dst": ValueRef.temporary("below_lower")},
                result_types={"dst": mask_pattern},
            ),
            _op_emit(
                descriptor=select_descriptor,
                operands={
                    "condition": ValueRef.temporary("below_lower"),
                    "true_value": ValueRef.operand("lower"),
                    "false_value": ValueRef.operand("value"),
                },
                results={"dst": ValueRef.temporary("lower_clamped")},
                result_types={"dst": type_pattern},
            ),
            _op_emit(
                descriptor=upper_compare_descriptor,
                operands={
                    "lhs": ValueRef.temporary("lower_clamped"),
                    "rhs": ValueRef.operand("upper"),
                },
                results={"dst": ValueRef.temporary("above_upper")},
                result_types={"dst": mask_pattern},
            ),
            _op_emit(
                descriptor=select_descriptor,
                operands={
                    "condition": ValueRef.temporary("above_upper"),
                    "true_value": ValueRef.operand("upper"),
                    "false_value": ValueRef.temporary("lower_clamped"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _scalar_clampf_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for type_pattern, suffix in ((_F32, "f32"), (_F64, "f64")):
        rules.append(
            _clampf_number_rule(
                scalar_arithmetic.scalar_clampf,
                type_pattern,
                f"llvmir.maxnum.{suffix}",
                f"llvmir.minnum.{suffix}",
            )
        )
        rules.append(
            _clampf_ordered_rule(
                scalar_arithmetic.scalar_clampf,
                type_pattern,
                _I1,
                f"llvmir.cmp.olt.{suffix}",
                f"llvmir.cmp.ogt.{suffix}",
                f"llvmir.select.{suffix}",
            )
        )
    return tuple(rules)


def _vector_clampf_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for lane_count in _VECTOR_LANE_COUNTS:
        type_pattern = _vector_type("f32", lane_count)
        mask_pattern = _vector_type("i1", lane_count)
        suffix = _vector_suffix("f32", lane_count)
        rules.append(
            _clampf_number_rule(
                vector.vector_clampf,
                type_pattern,
                f"llvmir.maxnum.{suffix}",
                f"llvmir.minnum.{suffix}",
            )
        )
        rules.append(
            _clampf_ordered_rule(
                vector.vector_clampf,
                type_pattern,
                mask_pattern,
                f"llvmir.cmp.olt.{suffix}",
                f"llvmir.cmp.ogt.{suffix}",
                f"llvmir.select.{suffix}",
            )
        )
    type_pattern = _vector_type("f32", 1)
    mask_pattern = _vector_type("i1", 1)
    rules.append(
        _clampf_number_rule(
            vector.vector_clampf,
            type_pattern,
            "llvmir.maxnum.f32",
            "llvmir.minnum.f32",
        )
    )
    rules.append(
        _clampf_ordered_rule(
            vector.vector_clampf,
            type_pattern,
            mask_pattern,
            "llvmir.cmp.olt.f32",
            "llvmir.cmp.ogt.f32",
            "llvmir.select.f32",
        )
    )
    return tuple(rules)


def _clampf_rules() -> tuple[DescriptorRule, ...]:
    return (*_scalar_clampf_rules(), *_vector_clampf_rules())


def _vector_constant_rules() -> tuple[DescriptorRule, ...]:
    return (
        tuple(
            _vector_const_i_rule("i32", lane_count, minimum=_I32_MIN, maximum=_I32_MAX)
            for lane_count in _VECTOR_LANE_COUNTS
        )
        + tuple(
            _vector_const_i_rule("i64", lane_count)
            for lane_count in _VECTOR_LANE_COUNTS
        )
        + tuple(
            _vector_const_float_rule(element, lane_count)
            for element in ("f16", "bf16", "f32", "f64")
            for lane_count in _VECTOR_LANE_COUNTS
        )
        + (
            _const_i32_rule(vector.vector_constant, _vector_type("i32", 1)),
            _const_i64_rule(vector.vector_constant, _vector_type("i64", 1)),
            _const_float_rule(
                vector.vector_constant,
                _vector_type("f16", 1),
                "llvmir.const.f16",
            ),
            _const_float_rule(
                vector.vector_constant,
                _vector_type("bf16", 1),
                "llvmir.const.bf16",
            ),
            _const_float_rule(
                vector.vector_constant,
                _vector_type("f32", 1),
                "llvmir.const.f32",
            ),
            _const_float_rule(
                vector.vector_constant,
                _vector_type("f64", 1),
                "llvmir.const.f64",
            ),
        )
    )


def _one_lane_splat_rule(element: str) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=vector.vector_splat,
        source=ValueRef.operand("scalar"),
        result=ValueRef.result("result"),
        guards=(
            Guard.value_type("scalar", Scalar(element)),
            Guard.value_type("result", _vector_type(element, 1)),
        ),
    )


def _splat_rule(element: str, lane_count: int) -> DescriptorRule:
    scalar_type = Scalar(element)
    result_type = _vector_type(element, lane_count)
    descriptor = _descriptor(f"llvmir.splat.{_vector_suffix(element, lane_count)}")
    return DescriptorRule(
        source_op=vector.vector_splat,
        descriptor=descriptor,
        guards=(
            Guard.value_type("scalar", scalar_type),
            Guard.value_type("result", result_type),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"value": ValueRef.operand("scalar")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _one_lane_from_elements_rule(element: str) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=vector.vector_from_elements,
        source=ValueRef.operand("elements", element=0),
        result=ValueRef.result("result"),
        guards=(
            Guard.operand_segment_count("elements", 1),
            Guard.value_type("elements", Scalar(element)),
            Guard.value_type("result", _vector_type(element, 1)),
        ),
    )


def _from_elements_rule(element: str, lane_count: int) -> DescriptorRule:
    scalar_type = Scalar(element)
    result_type = _vector_type(element, lane_count)
    descriptor = _descriptor(
        f"llvmir.from_elements.{_vector_suffix(element, lane_count)}"
    )
    return DescriptorRule(
        source_op=vector.vector_from_elements,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("elements", lane_count),
            Guard.value_type("elements", scalar_type),
            Guard.value_type("result", result_type),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    f"lane{lane_index}": ValueRef.operand(
                        "elements", element=lane_index
                    )
                    for lane_index in range(lane_count)
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _one_lane_extract_rule(element: str) -> ValueAliasRule:
    return ValueAliasRule(
        source_op=vector.vector_extract,
        source=ValueRef.operand("source"),
        result=ValueRef.result("result"),
        guards=(
            Guard.value_type("source", _vector_type(element, 1)),
            Guard.value_type("result", Scalar(element)),
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range(
                "static_indices", element=0, minimum=0, maximum=0
            ),
        ),
    )


def _extract_rule(element: str, lane_count: int) -> DescriptorRule:
    source_type = _vector_type(element, lane_count)
    result_type = Scalar(element)
    descriptor = _descriptor(f"llvmir.extract.{_vector_suffix(element, lane_count)}")
    return DescriptorRule(
        source_op=vector.vector_extract,
        descriptor=descriptor,
        guards=(
            Guard.value_type("source", source_type),
            Guard.value_type("result", result_type),
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range(
                "static_indices", element=0, minimum=0, maximum=lane_count - 1
            ),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"source": ValueRef.operand("source")},
                results={"dst": ValueRef.result("result")},
                immediates={
                    "lane": AttrProject.i64_array_element("static_indices", element=0)
                },
            ),
        ),
    )


def _one_lane_insert_rule(element: str) -> ValueAliasRule:
    vector_type = _vector_type(element, 1)
    return ValueAliasRule(
        source_op=vector.vector_insert,
        source=ValueRef.operand("value"),
        result=ValueRef.result("result"),
        guards=(
            Guard.value_type("value", Scalar(element)),
            Guard.value_type("dest", vector_type),
            Guard.value_type("result", vector_type),
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range(
                "static_indices", element=0, minimum=0, maximum=0
            ),
        ),
    )


def _insert_rule(element: str, lane_count: int) -> DescriptorRule:
    value_type = Scalar(element)
    dest_type = _vector_type(element, lane_count)
    descriptor = _descriptor(f"llvmir.insert.{_vector_suffix(element, lane_count)}")
    return DescriptorRule(
        source_op=vector.vector_insert,
        descriptor=descriptor,
        guards=(
            Guard.value_type("value", value_type),
            Guard.value_type("dest", dest_type),
            Guard.value_type("result", dest_type),
            Guard.operand_segment_count("indices", 0),
            Guard.i64_array_count("static_indices", 1),
            Guard.i64_array_element_range(
                "static_indices", element=0, minimum=0, maximum=lane_count - 1
            ),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={
                    "dest": ValueRef.operand("dest"),
                    "value": ValueRef.operand("value"),
                },
                results={"dst": ValueRef.result("result")},
                immediates={
                    "lane": AttrProject.i64_array_element("static_indices", element=0)
                },
            ),
        ),
    )


def _shuffle_rule(element: str, lane_count: int) -> DescriptorRule:
    type_pattern = _vector_type(element, lane_count)
    descriptor = _descriptor(f"llvmir.shuffle.{_vector_suffix(element, lane_count)}")
    return DescriptorRule(
        source_op=vector.vector_shuffle,
        descriptor=descriptor,
        guards=(
            Guard.value_type("source", type_pattern),
            Guard.value_type("result", type_pattern),
            Guard.i64_array_count("source_lanes", lane_count),
            Guard.i64_array_elements_range(
                "source_lanes", minimum=0, maximum=lane_count - 1
            ),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"source": ValueRef.operand("source")},
                results={"dst": ValueRef.result("result")},
                immediates={
                    f"lane{lane_index}": AttrProject.i64_array_element(
                        "source_lanes", element=lane_index
                    )
                    for lane_index in range(lane_count)
                },
            ),
        ),
    )


def _slice_rule(
    element: str,
    *,
    source_lane_count: int,
    result_lane_count: int,
) -> DescriptorRule:
    source_type = _vector_type(element, source_lane_count)
    result_type = _vector_type(element, result_lane_count)
    descriptor = _descriptor(
        "llvmir.slice."
        f"{_vector_suffix(element, source_lane_count)}."
        f"{_vector_suffix(element, result_lane_count)}"
    )
    return DescriptorRule(
        source_op=vector.vector_slice,
        descriptor=descriptor,
        guards=(
            Guard.value_type("source", source_type),
            Guard.value_type("result", result_type),
            Guard.operand_segment_count("offsets", 0),
            Guard.i64_array_count("static_offsets", 1),
            Guard.i64_array_element_range(
                "static_offsets",
                element=0,
                minimum=0,
                maximum=source_lane_count - result_lane_count,
            ),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"source": ValueRef.operand("source")},
                results={"dst": ValueRef.result("result")},
                immediates={
                    "offset": AttrProject.i64_array_element("static_offsets", element=0)
                },
            ),
        ),
    )


def _structural_vector_rules() -> tuple[DescriptorRule | ValueAliasRule, ...]:
    rules: list[DescriptorRule | ValueAliasRule] = []
    for element in _STRUCTURAL_VECTOR_TYPES:
        for lane_count in _VECTOR_LANE_COUNTS:
            rules.append(_splat_rule(element, lane_count))
            rules.append(_from_elements_rule(element, lane_count))
            rules.append(_extract_rule(element, lane_count))
            rules.append(_insert_rule(element, lane_count))
            rules.append(_shuffle_rule(element, lane_count))
        rules.append(_slice_rule(element, source_lane_count=4, result_lane_count=2))
        rules.append(_one_lane_splat_rule(element))
        rules.append(_one_lane_from_elements_rule(element))
        rules.append(_one_lane_extract_rule(element))
        rules.append(_one_lane_insert_rule(element))
    return tuple(rules)


def _memory_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for element, element_byte_count in _MEMORY_VALUE_TYPES:
        for source_dynamic_count, dynamic_term_count in (
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 2),
        ):
            rules.append(
                _view_load_rule(
                    view.view_load,
                    "view",
                    Scalar(element),
                    _memory_descriptor_key(
                        "load", element, dynamic=dynamic_term_count != 0
                    ),
                    source_dynamic_count=source_dynamic_count,
                    dynamic_term_count=dynamic_term_count,
                    element_byte_count=element_byte_count,
                )
            )
            rules.append(
                _view_store_rule(
                    view.view_store,
                    "view",
                    Scalar(element),
                    _memory_descriptor_key(
                        "store", element, dynamic=dynamic_term_count != 0
                    ),
                    source_dynamic_count=source_dynamic_count,
                    dynamic_term_count=dynamic_term_count,
                    element_byte_count=element_byte_count,
                )
            )
            for lane_count in _VECTOR_LANE_COUNTS:
                vector_type = _vector_type(element, lane_count)
                vector_suffix = _vector_suffix(element, lane_count)
                rules.append(
                    _view_load_rule(
                        vector.vector_load,
                        "view",
                        vector_type,
                        _memory_descriptor_key(
                            "load", vector_suffix, dynamic=dynamic_term_count != 0
                        ),
                        source_dynamic_count=source_dynamic_count,
                        dynamic_term_count=dynamic_term_count,
                        element_byte_count=element_byte_count,
                        vector_lane_count=lane_count,
                    )
                )
                rules.append(
                    _view_store_rule(
                        vector.vector_store,
                        "view",
                        vector_type,
                        _memory_descriptor_key(
                            "store", vector_suffix, dynamic=dynamic_term_count != 0
                        ),
                        source_dynamic_count=source_dynamic_count,
                        dynamic_term_count=dynamic_term_count,
                        element_byte_count=element_byte_count,
                        vector_lane_count=lane_count,
                    )
                )
            vector_type = _vector_type(element, 1)
            rules.append(
                _view_load_rule(
                    vector.vector_load,
                    "view",
                    vector_type,
                    _memory_descriptor_key(
                        "load", element, dynamic=dynamic_term_count != 0
                    ),
                    source_dynamic_count=source_dynamic_count,
                    dynamic_term_count=dynamic_term_count,
                    element_byte_count=element_byte_count,
                )
            )
            rules.append(
                _view_store_rule(
                    vector.vector_store,
                    "view",
                    vector_type,
                    _memory_descriptor_key(
                        "store", element, dynamic=dynamic_term_count != 0
                    ),
                    source_dynamic_count=source_dynamic_count,
                    dynamic_term_count=dynamic_term_count,
                    element_byte_count=element_byte_count,
                )
            )
    return tuple(rules)


def _kernel_query_rule(source_op: Op, query: str, dimension: str) -> DescriptorRule:
    descriptor = _descriptor(f"llvmir.kernel.{query}.{dimension}")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("dimension", dimension),
            Guard.value_type("result", _INDEX),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _kernel_query_rules() -> tuple[DescriptorRule, ...]:
    source_ops = (
        (kernel.kernel_workitem_id, "workitem_id"),
        (kernel.kernel_workgroup_id, "workgroup_id"),
        (kernel.kernel_workgroup_size, "workgroup_size"),
        (kernel.kernel_workitem_dispatch_id, "workitem_dispatch_id"),
    )
    return tuple(
        _kernel_query_rule(source_op, query, dimension)
        for source_op, query in source_ops
        for dimension in _KERNEL_DIMENSIONS
    )


LLVMIR_GENERIC_CORE_CONTRACT_DIALECT_OPS = {
    "buffer": ALL_BUFFER_OPS,
    "index": ALL_INDEX_OPS,
    "kernel": ALL_KERNEL_OPS,
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
        _const_i64_rule(index.index_constant, _INDEX),
        _const_i64_rule(scalar_conversion.scalar_constant, _I64),
        _const_i64_rule(index.index_constant, _OFFSET),
        _const_float_rule(
            scalar_conversion.scalar_constant,
            _F16,
            "llvmir.const.f16",
        ),
        _const_float_rule(
            scalar_conversion.scalar_constant,
            _BF16,
            "llvmir.const.bf16",
        ),
        _const_float_rule(
            scalar_conversion.scalar_constant,
            _F32,
            "llvmir.const.f32",
        ),
        _const_float_rule(
            scalar_conversion.scalar_constant,
            _F64,
            "llvmir.const.f64",
        ),
        ValueAliasRule(
            source_op=index.index_assume,
            source=ValueRef.operand("values"),
            result=ValueRef.result("results"),
            guards=(Guard.operand_segment_count("values", 1),),
        ),
        _buffer_alloca_rule("private"),
        _buffer_alloca_rule("workgroup"),
        _buffer_view_rule(),
        *_scalar_arithmetic_rules(),
        *_scalar_bitwise_rules(),
        _binary_rule(index.index_add, _INDEX, "llvmir.add.i64"),
        _binary_rule(index.index_sub, _INDEX, "llvmir.sub.i64"),
        _binary_rule(index.index_mul, _INDEX, "llvmir.mul.i64"),
        _binary_rule(index.index_div, _INDEX, "llvmir.udiv.i64"),
        _binary_rule(index.index_rem, _INDEX, "llvmir.urem.i64"),
        _index_madd_rule(),
        *_index_minmax_rules(),
        *_index_bitwise_rules(),
        _binary_rule(index.index_add, _OFFSET, "llvmir.add.i64"),
        _binary_rule(index.index_sub, _OFFSET, "llvmir.sub.i64"),
        *_vector_arithmetic_rules(),
        *_vector_bitwise_rules(),
        *_clampf_rules(),
        *_compare_rules(),
        *_cast_rules(),
        *_index_cast_rules(),
        *_select_rules(),
        *_kernel_query_rules(),
        *_vector_constant_rules(),
        *_structural_vector_rules(),
        *_memory_rules(),
    ),
)
