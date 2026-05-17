# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SPIR-V logical source-to-low contract fragment."""

from __future__ import annotations

from loom.dialect.buffer import ALL_BUFFER_OPS
from loom.dialect.buffer import defs as buffer
from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.index import defs as index
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scf import ALL_SCF_OPS
from loom.dialect.scf import defs as scf
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dialect.view import ALL_VIEW_OPS
from loom.dialect.view import defs as view
from loom.dsl import Op
from loom.error.target import ERR_TARGET_050
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
from loom.target.arch.spirv.scalar_alu import (
    FLOAT_BINARY_OPERATIONS,
    FLOAT_SCALAR_ALU_TYPES,
    INTEGER_SCALAR_ALU_TYPE_PAIRS,
    OFFSET64_ALU_TYPE,
    OFFSET64_COMPARE_PREDICATES,
    SCALAR_ALU_TYPES,
    SIGNED_INTEGER_BINARY_OPERATIONS,
    SIGNED_INTEGER_COMPARE_PREDICATES,
    SIGNED_INTEGER_SCALAR_ALU_TYPES,
    UNSIGNED_INTEGER_BINARY_OPERATIONS,
    UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES,
    IntegerAluTypePair,
    IntegerComparePredicate,
    ScalarAluType,
    ScalarBinaryOperation,
)
from loom.target.arch.spirv.scalar_conversion import (
    DIRECT_SCALAR_CONVERSIONS,
    UNSIGNED_SCALAR_CONVERSIONS,
    ScalarConversion,
)
from loom.target.arch.spirv.scalar_memory import (
    STORAGE_BUFFER_SCALARS,
    StorageBufferScalar,
)
from loom.target.contracts import (
    AttrProject,
    ContractFragment,
    DescriptorEmitForm,
    DescriptorMatrixRule,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    Scalar,
    SourceMemoryConstraint,
    SourceMemoryDynamicIndexSource,
    SourceMemoryOperation,
    SourceMemoryRootKind,
    TypePattern,
    ValueAliasRule,
    ValueRef,
    View,
    descriptor_by_key,
    source_memory_minimum_alignment_param,
    target_diagnostic,
    u32_param,
    value_type_param,
)
from loom.target.low_descriptors import Descriptor

_I1 = Scalar("i1")
_I32 = Scalar("i32")
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(SPIRV_LOGICAL_CORE_DESCRIPTOR_SET, key)


def _typed_guards(
    fields: tuple[str, ...],
    type_pattern: TypePattern,
) -> tuple[Guard, ...]:
    return tuple(Guard.value_type(field, type_pattern) for field in fields)


def _feature_guards(descriptor: Descriptor) -> tuple[Guard, ...]:
    return (
        (Guard.descriptor_available(descriptor),)
        if descriptor.feature_mask_words
        else ()
    )


def _descriptor_emit(
    *,
    descriptor: Descriptor,
    operands: dict[str, ValueRef] | None = None,
    results: dict[str, ValueRef] | None = None,
    immediates: dict[str, AttrProject] | None = None,
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


def _i32_index_constant_rule() -> DescriptorRule:
    descriptor = _descriptor("spirv.op_constant.i32")
    return DescriptorRule(
        source_op=index.index_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64"),
            Guard.value_type("result", _INDEX),
            Guard.i64_range("value", -(2**31), (2**31) - 1),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"i32_value": AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _offset_constant_rule() -> DescriptorRule:
    descriptor = _descriptor("spirv.op_constant.offset64")
    return DescriptorRule(
        source_op=index.index_constant,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64"),
            Guard.value_type("result", _OFFSET),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"offset64_value": AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _i32_constant_rule(source_op: Op) -> DescriptorRule:
    descriptor = _descriptor("spirv.op_constant.i32")
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.attr_kind("value", "i64"),
            Guard.value_type("result", _I32),
            Guard.i64_range("value", -(2**31), (2**31) - 1),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                results={"dst": ValueRef.result("result")},
                immediates={"i32_value": AttrProject.direct("value")},
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
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), type_pattern),
            *_feature_guards(descriptor),
        ),
        emit=(
            _descriptor_emit(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
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
        guards=(
            *_typed_guards(("a", "b", "c", "result"), type_pattern),
            *_feature_guards(descriptor),
        ),
        emit=(
            _descriptor_emit(
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


def _conversion_rule(row: ScalarConversion) -> DescriptorRule:
    descriptor = _descriptor(row.key)
    return DescriptorRule(
        source_op=_CONVERSION_SOURCE_OPS[row.source_op_key],
        descriptor=descriptor,
        guards=(
            Guard.value_type("input", Scalar(row.source_type.source_type)),
            Guard.value_type("result", Scalar(row.result_type.source_type)),
            *_feature_guards(descriptor),
        ),
        emit=(
            _descriptor_emit(
                descriptor=descriptor,
                operands={"input": ValueRef.operand("input")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _integer_view_key(source_type: ScalarAluType, result_type: ScalarAluType) -> str:
    return f"spirv.op_bitcast.{source_type.suffix}.{result_type.suffix}"


def _integer_view_emit(
    *,
    source_type: ScalarAluType,
    result_type: ScalarAluType,
    input_ref: ValueRef,
    output_ref: ValueRef,
) -> EmitDescriptorOp:
    return _descriptor_emit(
        descriptor=_descriptor(_integer_view_key(source_type, result_type)),
        operands={"input": input_ref},
        results={"dst": output_ref},
    )


def _unsigned_binary_rule(
    scalar_pair: IntegerAluTypePair,
    operation: ScalarBinaryOperation,
) -> DescriptorRule:
    descriptor = _descriptor(
        f"spirv.op_{operation.descriptor_suffix}.{scalar_pair.unsigned.suffix}"
    )
    type_pattern = Scalar(scalar_pair.source_type)
    return DescriptorRule(
        source_op=_INTEGER_BINARY_SOURCE_OPS[operation.source_op_key],
        descriptor=descriptor,
        guards=(
            *_typed_guards(("lhs", "rhs", "result"), type_pattern),
            *_feature_guards(descriptor),
        ),
        emit=(
            _integer_view_emit(
                source_type=scalar_pair.signed,
                result_type=scalar_pair.unsigned,
                input_ref=ValueRef.operand("lhs"),
                output_ref=ValueRef.temporary("unsigned_lhs"),
            ),
            _integer_view_emit(
                source_type=scalar_pair.signed,
                result_type=scalar_pair.unsigned,
                input_ref=ValueRef.operand("rhs"),
                output_ref=ValueRef.temporary("unsigned_rhs"),
            ),
            _descriptor_emit(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.temporary("unsigned_lhs"),
                    "rhs": ValueRef.temporary("unsigned_rhs"),
                },
                results={"dst": ValueRef.temporary("unsigned_result")},
            ),
            _integer_view_emit(
                source_type=scalar_pair.unsigned,
                result_type=scalar_pair.signed,
                input_ref=ValueRef.temporary("unsigned_result"),
                output_ref=ValueRef.result("result"),
            ),
        ),
    )


def _unsigned_conversion_rule(row: ScalarConversion) -> DescriptorRule:
    descriptor = _descriptor(row.key)
    source_op = _CONVERSION_SOURCE_OPS[row.source_op_key]
    if row.source_op_key == "extui":
        source_pair = _INTEGER_ALU_TYPE_PAIR_BY_SOURCE_TYPE[row.source_type.source_type]
        result_pair = _INTEGER_ALU_TYPE_PAIR_BY_SOURCE_TYPE[row.result_type.source_type]
        return DescriptorRule(
            source_op=source_op,
            descriptor=descriptor,
            guards=(
                Guard.value_type("input", Scalar(row.source_type.source_type)),
                Guard.value_type("result", Scalar(row.result_type.source_type)),
                *_feature_guards(descriptor),
            ),
            emit=(
                _integer_view_emit(
                    source_type=source_pair.signed,
                    result_type=source_pair.unsigned,
                    input_ref=ValueRef.operand("input"),
                    output_ref=ValueRef.temporary("unsigned_input"),
                ),
                _descriptor_emit(
                    descriptor=descriptor,
                    operands={"input": ValueRef.temporary("unsigned_input")},
                    results={"dst": ValueRef.temporary("unsigned_result")},
                ),
                _integer_view_emit(
                    source_type=result_pair.unsigned,
                    result_type=result_pair.signed,
                    input_ref=ValueRef.temporary("unsigned_result"),
                    output_ref=ValueRef.result("result"),
                ),
            ),
        )
    if row.source_op_key == "uitofp":
        source_pair = _INTEGER_ALU_TYPE_PAIR_BY_SOURCE_TYPE[row.source_type.source_type]
        return DescriptorRule(
            source_op=source_op,
            descriptor=descriptor,
            guards=(
                Guard.value_type("input", Scalar(row.source_type.source_type)),
                Guard.value_type("result", Scalar(row.result_type.source_type)),
                *_feature_guards(descriptor),
            ),
            emit=(
                _integer_view_emit(
                    source_type=source_pair.signed,
                    result_type=source_pair.unsigned,
                    input_ref=ValueRef.operand("input"),
                    output_ref=ValueRef.temporary("unsigned_input"),
                ),
                _descriptor_emit(
                    descriptor=descriptor,
                    operands={"input": ValueRef.temporary("unsigned_input")},
                    results={"dst": ValueRef.result("result")},
                ),
            ),
        )
    if row.source_op_key == "fptoui":
        result_pair = _INTEGER_ALU_TYPE_PAIR_BY_SOURCE_TYPE[row.result_type.source_type]
        return DescriptorRule(
            source_op=source_op,
            descriptor=descriptor,
            guards=(
                Guard.value_type("input", Scalar(row.source_type.source_type)),
                Guard.value_type("result", Scalar(row.result_type.source_type)),
                *_feature_guards(descriptor),
            ),
            emit=(
                _descriptor_emit(
                    descriptor=descriptor,
                    operands={"input": ValueRef.operand("input")},
                    results={"dst": ValueRef.temporary("unsigned_result")},
                ),
                _integer_view_emit(
                    source_type=result_pair.unsigned,
                    result_type=result_pair.signed,
                    input_ref=ValueRef.temporary("unsigned_result"),
                    output_ref=ValueRef.result("result"),
                ),
            ),
        )
    raise ValueError(f"unsupported unsigned SPIR-V scalar conversion {row.key}")


def _compare_rule(
    source_op: Op,
    predicate: IntegerComparePredicate,
    type_pattern: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate.source_predicate),
            *_typed_guards(("lhs", "rhs"), type_pattern),
            Guard.value_type("result", _I1),
            *_feature_guards(descriptor),
        ),
        emit=(
            _descriptor_emit(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _unsigned_compare_rule(
    source_op: Op,
    predicate: IntegerComparePredicate,
    scalar_pair: IntegerAluTypePair,
    type_pattern: TypePattern | None = None,
) -> DescriptorRule:
    descriptor = _descriptor(
        f"spirv.op_{predicate.descriptor_suffix}.{scalar_pair.unsigned.suffix}"
    )
    operand_type_pattern = (
        Scalar(scalar_pair.source_type) if type_pattern is None else type_pattern
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("predicate", predicate.source_predicate),
            *_typed_guards(("lhs", "rhs"), operand_type_pattern),
            Guard.value_type("result", _I1),
            *_feature_guards(descriptor),
        ),
        emit=(
            _integer_view_emit(
                source_type=scalar_pair.signed,
                result_type=scalar_pair.unsigned,
                input_ref=ValueRef.operand("lhs"),
                output_ref=ValueRef.temporary("unsigned_lhs"),
            ),
            _integer_view_emit(
                source_type=scalar_pair.signed,
                result_type=scalar_pair.unsigned,
                input_ref=ValueRef.operand("rhs"),
                output_ref=ValueRef.temporary("unsigned_rhs"),
            ),
            _descriptor_emit(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.temporary("unsigned_lhs"),
                    "rhs": ValueRef.temporary("unsigned_rhs"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _select_rule(
    type_pattern: TypePattern,
    descriptor_key: str,
) -> DescriptorRule:
    descriptor = _descriptor(descriptor_key)
    return DescriptorRule(
        source_op=scf.scf_select,
        descriptor=descriptor,
        guards=(
            Guard.value_type("condition", _I1),
            *_typed_guards(("true_value", "false_value", "result"), type_pattern),
            *_feature_guards(descriptor),
        ),
        emit=(
            _descriptor_emit(
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


def _buffer_view_rule(scalar: StorageBufferScalar) -> DescriptorRule:
    view_type = View(scalar.source_type)
    descriptor = _descriptor(
        f"spirv.op_ptr_access_chain.storage_buffer.{scalar.suffix}.byte_offset"
    )
    return DescriptorRule(
        source_op=buffer.buffer_view,
        descriptor=descriptor,
        guards=(
            Guard.value_type("result", view_type),
            *_feature_guards(descriptor),
        ),
        emit=(
            _descriptor_emit(
                descriptor=descriptor,
                operands={
                    "base": ValueRef.operand("buffer"),
                    "byte_offset": ValueRef.operand("byte_offset"),
                },
                results={"ptr": ValueRef.result("result")},
            ),
        ),
    )


_STORAGE_BUFFER_MEMORY_SPACES = ("unknown", "generic", "global", "descriptor")


def _storage_buffer_alignment_diagnostic(
    required_alignment: int,
) -> GuardDiagnostic:
    return GuardDiagnostic(
        ref=target_diagnostic(
            ERR_TARGET_050,
            value_type_param("value_type", "view"),
            u32_param("required_alignment", required_alignment),
            source_memory_minimum_alignment_param("known_alignment"),
        )
    )


def _storage_buffer_source_memory(
    operation: SourceMemoryOperation,
    scalar: StorageBufferScalar,
) -> SourceMemoryConstraint:
    return SourceMemoryConstraint(
        operation=operation,
        root_kind=SourceMemoryRootKind.ANY,
        memory_spaces=_STORAGE_BUFFER_MEMORY_SPACES,
        element_byte_count=scalar.byte_width,
        vector_lane_count=1,
        vector_lane_byte_stride=scalar.byte_width,
        static_byte_offset_minimum=-(2**63),
        static_byte_offset_maximum=(2**63) - 1,
        minimum_alignment=scalar.byte_width,
        dynamic_term_count=None,
        dynamic_index_source=SourceMemoryDynamicIndexSource.NONE,
        diagnostic=_storage_buffer_alignment_diagnostic(scalar.byte_width),
    )


def _view_load_rule(scalar: StorageBufferScalar) -> DescriptorRule:
    scalar_type = Scalar(scalar.source_type)
    view_type = View(scalar.source_type)
    descriptor = _descriptor(f"spirv.op_load.storage_buffer.{scalar.suffix}")
    return DescriptorRule(
        source_op=view.view_load,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("indices", 0),
            Guard.value_type("view", view_type),
            Guard.value_type("result", scalar_type),
            *_feature_guards(descriptor),
        ),
        emit=(
            _descriptor_emit(
                descriptor=descriptor,
                operands={"ptr": ValueRef.operand("view")},
                results={"dst": ValueRef.result("result")},
                source_memory=_storage_buffer_source_memory(
                    SourceMemoryOperation.LOAD,
                    scalar,
                ),
            ),
        ),
    )


def _view_store_rule(scalar: StorageBufferScalar) -> DescriptorRule:
    scalar_type = Scalar(scalar.source_type)
    view_type = View(scalar.source_type)
    descriptor = _descriptor(f"spirv.op_store.storage_buffer.{scalar.suffix}")
    return DescriptorRule(
        source_op=view.view_store,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("indices", 0),
            Guard.value_type("view", view_type),
            Guard.value_type("value", scalar_type),
            *_feature_guards(descriptor),
        ),
        emit=(
            _descriptor_emit(
                descriptor=descriptor,
                operands={
                    "ptr": ValueRef.operand("view"),
                    "value": ValueRef.operand("value"),
                },
                source_memory=_storage_buffer_source_memory(
                    SourceMemoryOperation.STORE,
                    scalar,
                ),
            ),
        ),
    )


def _storage_buffer_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for scalar in STORAGE_BUFFER_SCALARS:
        rules.append(_buffer_view_rule(scalar))
        rules.append(_view_load_rule(scalar))
        rules.append(_view_store_rule(scalar))
    return tuple(rules)


_INTEGER_BINARY_SOURCE_OPS = {
    "addi": scalar_arithmetic.scalar_addi,
    "subi": scalar_arithmetic.scalar_subi,
    "muli": scalar_arithmetic.scalar_muli,
    "divui": scalar_arithmetic.scalar_divui,
    "divsi": scalar_arithmetic.scalar_divsi,
    "remui": scalar_arithmetic.scalar_remui,
    "remsi": scalar_arithmetic.scalar_remsi,
}

_FLOAT_BINARY_SOURCE_OPS = {
    "addf": scalar_arithmetic.scalar_addf,
    "subf": scalar_arithmetic.scalar_subf,
    "mulf": scalar_arithmetic.scalar_mulf,
    "divf": scalar_arithmetic.scalar_divf,
    "remf": scalar_arithmetic.scalar_remf,
}

_CONVERSION_SOURCE_OPS = {
    "sitofp": scalar_conversion.scalar_sitofp,
    "uitofp": scalar_conversion.scalar_uitofp,
    "fptosi": scalar_conversion.scalar_fptosi,
    "fptoui": scalar_conversion.scalar_fptoui,
    "extf": scalar_conversion.scalar_extf,
    "fptrunc": scalar_conversion.scalar_fptrunc,
    "extsi": scalar_conversion.scalar_extsi,
    "extui": scalar_conversion.scalar_extui,
    "trunci": scalar_conversion.scalar_trunci,
    "bitcast": scalar_conversion.scalar_bitcast,
}

_INTEGER_ALU_TYPE_PAIR_BY_SOURCE_TYPE = {
    scalar_pair.source_type: scalar_pair
    for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS
}


def _scalar_type_pattern(scalar: ScalarAluType) -> TypePattern:
    return Scalar(scalar.source_type)


def _scalar_binary_rule(
    scalar: ScalarAluType,
    operation: ScalarBinaryOperation,
    source_ops: dict[str, Op],
) -> DescriptorRule:
    return _binary_rule(
        source_ops[operation.source_op_key],
        _scalar_type_pattern(scalar),
        f"spirv.op_{operation.descriptor_suffix}.{scalar.suffix}",
    )


def _scalar_binary_rules() -> tuple[DescriptorRule, ...]:
    rules = [
        _scalar_binary_rule(scalar, operation, _INTEGER_BINARY_SOURCE_OPS)
        for scalar in SIGNED_INTEGER_SCALAR_ALU_TYPES
        for operation in SIGNED_INTEGER_BINARY_OPERATIONS
    ]
    rules.extend(
        _unsigned_binary_rule(scalar_pair, operation)
        for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS
        for operation in UNSIGNED_INTEGER_BINARY_OPERATIONS
    )
    rules.extend(
        _scalar_binary_rule(scalar, operation, _FLOAT_BINARY_SOURCE_OPS)
        for scalar in FLOAT_SCALAR_ALU_TYPES
        for operation in FLOAT_BINARY_OPERATIONS
    )
    return tuple(rules)


def _conversion_rules() -> tuple[DescriptorRule, ...]:
    rules = [_conversion_rule(row) for row in DIRECT_SCALAR_CONVERSIONS]
    rules.extend(_unsigned_conversion_rule(row) for row in UNSIGNED_SCALAR_CONVERSIONS)
    return tuple(rules)


def _compare_rules() -> tuple[DescriptorRule, ...]:
    rules = [
        _compare_rule(
            scalar_comparison.scalar_cmpi,
            predicate,
            _scalar_type_pattern(scalar),
            f"spirv.op_{predicate.descriptor_suffix}.{scalar.suffix}",
        )
        for scalar in SIGNED_INTEGER_SCALAR_ALU_TYPES
        for predicate in SIGNED_INTEGER_COMPARE_PREDICATES
    ]
    rules.extend(
        _unsigned_compare_rule(
            scalar_comparison.scalar_cmpi,
            predicate,
            scalar_pair,
        )
        for scalar_pair in INTEGER_SCALAR_ALU_TYPE_PAIRS
        for predicate in UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES
    )
    rules.extend(
        _compare_rule(
            index.index_cmp,
            predicate,
            _INDEX,
            f"spirv.op_{predicate.descriptor_suffix}.i32",
        )
        for predicate in SIGNED_INTEGER_COMPARE_PREDICATES
    )
    rules.extend(
        _unsigned_compare_rule(
            index.index_cmp,
            predicate,
            _INTEGER_ALU_TYPE_PAIR_BY_SOURCE_TYPE["i32"],
            _INDEX,
        )
        for predicate in UNSIGNED_ORDERED_INTEGER_COMPARE_PREDICATES
    )
    rules.extend(
        _compare_rule(
            index.index_cmp,
            predicate,
            _OFFSET,
            f"spirv.op_{predicate.descriptor_suffix}.{OFFSET64_ALU_TYPE.suffix}",
        )
        for predicate in OFFSET64_COMPARE_PREDICATES
    )
    return tuple(rules)


def _select_rules() -> tuple[DescriptorRule, ...]:
    rules = [
        _select_rule(_scalar_type_pattern(scalar), f"spirv.op_select.{scalar.suffix}")
        for scalar in SCALAR_ALU_TYPES
    ]
    rules.append(_select_rule(_INDEX, "spirv.op_select.i32"))
    rules.append(_select_rule(_OFFSET, f"spirv.op_select.{OFFSET64_ALU_TYPE.suffix}"))
    return tuple(rules)


SPIRV_LOGICAL_CORE_CONTRACT_DIALECT_OPS = {
    "buffer": ALL_BUFFER_OPS,
    "index": ALL_INDEX_OPS,
    "scalar": ALL_SCALAR_OPS,
    "scf": ALL_SCF_OPS,
    "vector": ALL_VECTOR_OPS,
    "view": ALL_VIEW_OPS,
}

SPIRV_LOGICAL_CORE_CONTRACT_FRAGMENT = ContractFragment(
    name="spirv.logical.core",
    descriptor_set=SPIRV_LOGICAL_CORE_DESCRIPTOR_SET,
    public_header="loom/target/arch/spirv/contracts/logical_core.h",
    cases=(
        _i32_constant_rule(scalar_conversion.scalar_constant),
        _i32_index_constant_rule(),
        _offset_constant_rule(),
        ValueAliasRule(
            source_op=index.index_assume,
            source=ValueRef.operand("values"),
            result=ValueRef.result("results"),
            guards=(Guard.operand_segment_count("values", 1),),
        ),
        ValueAliasRule(
            source_op=buffer.buffer_assume_alignment,
            source=ValueRef.operand("buffers"),
            result=ValueRef.result("results"),
            guards=(Guard.operand_segment_count("buffers", 1),),
        ),
        *_conversion_rules(),
        *_scalar_binary_rules(),
        _binary_rule(index.index_add, _INDEX, "spirv.op_iadd.i32"),
        _binary_rule(index.index_sub, _INDEX, "spirv.op_isub.i32"),
        _binary_rule(index.index_mul, _INDEX, "spirv.op_imul.i32"),
        _ternary_rule(index.index_madd, _INDEX, "spirv.op_imul_add.i32"),
        _binary_rule(index.index_add, _OFFSET, "spirv.op_iadd.offset64"),
        _binary_rule(index.index_sub, _OFFSET, "spirv.op_isub.offset64"),
        *_compare_rules(),
        *_select_rules(),
        *_storage_buffer_rules(),
        DescriptorMatrixRule(
            source_op=vector.vector_mma,
            source="vector_mma",
        ),
    ),
)
