# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TileLang/TIR type conversion helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from loom.ir import (
    BF16,
    F16,
    F32,
    F64,
    I1,
    I8,
    I16,
    I32,
    I64,
    INDEX,
    OFFSET,
    DynamicDim,
    ScalarType,
    ShapedType,
    StaticDim,
    Type,
    TypeKind,
)


class TileLangTypeConverter:
    """Maps TileLang/TIR dtypes and buffers to Loom types."""

    def map_dtype(self, dtype: Any, *, index_like: bool = False) -> Type:
        if index_like:
            return INDEX
        text = str(dtype)
        mapped = _DTYPE_MAP.get(text)
        if mapped is None:
            mapped = _parse_vector_dtype(text)
        if mapped is None:
            raise ValueError(f"unsupported TileLang dtype `{text}`")
        return mapped

    def vector_type(self, dtype: Any, lanes: int) -> ShapedType:
        element_type = dtype if isinstance(dtype, ScalarType) else self.map_dtype(dtype)
        if not isinstance(element_type, ScalarType):
            raise ValueError(f"vector element dtype must be scalar, got {dtype!r}")
        return ShapedType(TypeKind.VECTOR, element_type, (StaticDim(lanes),))

    def view_type(self, buffer: object) -> ShapedType:
        dtype = _attribute(buffer, "dtype")
        element_type = self.map_dtype(dtype)
        if not isinstance(element_type, ScalarType):
            raise ValueError(f"buffer element dtype must be scalar, got {dtype!r}")
        shape = _attribute(buffer, "shape", ())
        if shape is None:
            shape = ()
        if not isinstance(shape, Iterable) or isinstance(shape, str | bytes):
            raise ValueError(f"buffer shape must be iterable, got {shape!r}")
        dims = tuple(_shape_dim(dim) for dim in shape)
        return ShapedType(TypeKind.VIEW, element_type, dims)

    def buffer_byte_length(self, buffer: object) -> int | None:
        view_type = self.view_type(buffer)
        if not view_type.is_all_static:
            return None
        element_size = _ELEMENT_BYTE_SIZES.get(str(view_type.element_type))
        if element_size is None:
            return None
        byte_length = element_size
        for dim in view_type.dims:
            if not isinstance(dim, StaticDim):
                return None
            byte_length *= dim.size
        return byte_length

    def buffer_base_alignment(self, buffer: object) -> int:
        element_type = self.view_type(buffer).element_type
        return _ELEMENT_BYTE_SIZES.get(str(element_type), 1)


def _shape_dim(value: object) -> StaticDim | DynamicDim:
    if isinstance(value, int):
        return StaticDim(value)
    payload = getattr(value, "value", None)
    if isinstance(payload, int):
        return StaticDim(payload)
    text = str(value)
    if text.isdecimal():
        return StaticDim(int(text))
    return DynamicDim()


def _attribute(value: object, name: str, default: object | None = None) -> object:
    return getattr(value, name, default)


def _parse_vector_dtype(text: str) -> ShapedType | None:
    head, separator, tail = text.rpartition("x")
    if not separator or not tail.isdecimal():
        return None
    element_type = _DTYPE_MAP.get(head)
    if not isinstance(element_type, ScalarType):
        return None
    return ShapedType(TypeKind.VECTOR, element_type, (StaticDim(int(tail)),))


_DTYPE_MAP: dict[str, Type] = {
    "bool": I1,
    "int8": I8,
    "uint8": I8,
    "int16": I16,
    "uint16": I16,
    "int32": I32,
    "uint32": I32,
    "int64": I64,
    "uint64": I64,
    "index": INDEX,
    "offset": OFFSET,
    "float16": F16,
    "float32": F32,
    "float64": F64,
    "bfloat16": BF16,
}

_ELEMENT_BYTE_SIZES: dict[str, int] = {
    "i1": 1,
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "i64": 8,
    "index": 8,
    "offset": 8,
    "f16": 2,
    "bf16": 2,
    "f32": 4,
    "f64": 8,
}
