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
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
from loom.target.arch.spirv.scalar_alu import (
    FLOAT_BINARY_OPERATIONS,
    FLOAT_SCALAR_ALU_TYPES,
    INTEGER_BINARY_OPERATIONS,
    INTEGER_COMPARE_PREDICATES,
    INTEGER_SCALAR_ALU_TYPES,
    OFFSET64_ALU_TYPE,
    SCALAR_ALU_TYPES,
    UNSIGNED_INTEGER_COMPARE_PREDICATES,
    IntegerComparePredicate,
    ScalarAluType,
    ScalarBinaryOperation,
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
    Scalar,
    TypePattern,
    ValueRef,
    View,
    descriptor_by_key,
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
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor,
        operands={} if operands is None else operands,
        results={} if results is None else results,
        immediates={} if immediates is None else immediates,
        form=DescriptorEmitForm.OP,
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
        for scalar in INTEGER_SCALAR_ALU_TYPES
        for operation in INTEGER_BINARY_OPERATIONS
    ]
    rules.extend(
        _scalar_binary_rule(scalar, operation, _FLOAT_BINARY_SOURCE_OPS)
        for scalar in FLOAT_SCALAR_ALU_TYPES
        for operation in FLOAT_BINARY_OPERATIONS
    )
    return tuple(rules)


def _compare_rules() -> tuple[DescriptorRule, ...]:
    rules = [
        _compare_rule(
            scalar_comparison.scalar_cmpi,
            predicate,
            _scalar_type_pattern(scalar),
            f"spirv.op_{predicate.descriptor_suffix}.{scalar.suffix}",
        )
        for scalar in INTEGER_SCALAR_ALU_TYPES
        for predicate in INTEGER_COMPARE_PREDICATES
    ]
    rules.extend(
        _compare_rule(
            index.index_cmp,
            predicate,
            _INDEX,
            f"spirv.op_{predicate.descriptor_suffix}.i32",
        )
        for predicate in INTEGER_COMPARE_PREDICATES
    )
    rules.extend(
        _compare_rule(
            index.index_cmp,
            predicate,
            _OFFSET,
            f"spirv.op_{predicate.descriptor_suffix}.{OFFSET64_ALU_TYPE.suffix}",
        )
        for predicate in UNSIGNED_INTEGER_COMPARE_PREDICATES
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
