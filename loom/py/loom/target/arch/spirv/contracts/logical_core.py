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
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.view import ALL_VIEW_OPS
from loom.dialect.view import defs as view
from loom.dsl import Op
from loom.target.arch.spirv.descriptors import SPIRV_LOGICAL_CORE_DESCRIPTOR_SET
from loom.target.contracts import (
    AttrProject,
    ContractFragment,
    DescriptorEmitForm,
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

_I32 = Scalar("i32")
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")
_I32_VIEW = View("i32")


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(SPIRV_LOGICAL_CORE_DESCRIPTOR_SET, key)


def _typed_guards(
    fields: tuple[str, ...],
    type_pattern: TypePattern,
) -> tuple[Guard, ...]:
    return tuple(Guard.value_type(field, type_pattern) for field in fields)


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
        guards=_typed_guards(("lhs", "rhs", "result"), type_pattern),
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


def _buffer_view_rule() -> DescriptorRule:
    descriptor = _descriptor("spirv.op_ptr_access_chain.storage_buffer.byte_offset")
    return DescriptorRule(
        source_op=buffer.buffer_view,
        descriptor=descriptor,
        guards=(Guard.value_type("result", _I32_VIEW),),
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


def _view_load_i32_rule() -> DescriptorRule:
    descriptor = _descriptor("spirv.op_load.storage_buffer.i32")
    return DescriptorRule(
        source_op=view.view_load,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("indices", 0),
            Guard.value_type("view", _I32_VIEW),
            Guard.value_type("result", _I32),
        ),
        emit=(
            _descriptor_emit(
                descriptor=descriptor,
                operands={"ptr": ValueRef.operand("view")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _view_store_i32_rule() -> DescriptorRule:
    descriptor = _descriptor("spirv.op_store.storage_buffer.i32")
    return DescriptorRule(
        source_op=view.view_store,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("indices", 0),
            Guard.value_type("view", _I32_VIEW),
            Guard.value_type("value", _I32),
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


SPIRV_LOGICAL_CORE_CONTRACT_DIALECT_OPS = {
    "buffer": ALL_BUFFER_OPS,
    "index": ALL_INDEX_OPS,
    "scalar": ALL_SCALAR_OPS,
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
        _binary_rule(scalar_arithmetic.scalar_addi, _I32, "spirv.op_iadd.i32"),
        _binary_rule(index.index_add, _INDEX, "spirv.op_iadd.i32"),
        _binary_rule(index.index_add, _OFFSET, "spirv.op_iadd.offset64"),
        _buffer_view_rule(),
        _view_load_i32_rule(),
        _view_store_i32_rule(),
    ),
)
