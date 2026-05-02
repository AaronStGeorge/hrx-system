# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TileLang-specific import session state."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from loom.builder import ValueRef
from loom.importers.core import SourceImportSession
from loom.importers.tilelang.nodes import dtype, node_kind, source_name
from loom.importers.tilelang.types import TileLangTypeConverter
from loom.ir import (
    ENCODING_LAYOUT_TYPE,
    INDEX,
    OFFSET,
    DynamicEncoding,
    ShapedType,
    Type,
)


@dataclass(slots=True)
class TileLangConversionContext(SourceImportSession):
    """TileLang-specialized import session using Loom dynamic builders."""

    type_converter: TileLangTypeConverter = field(default_factory=TileLangTypeConverter)
    index_values: dict[object, ValueRef] = field(default_factory=dict)
    semantic_values: dict[tuple[object, ...], ValueRef] = field(default_factory=dict)
    semantic_value_types: dict[tuple[object, ...], str] = field(default_factory=dict)
    buffer_data_values: dict[tuple[object, ...], ValueRef] = field(default_factory=dict)
    semantic_index_values: dict[tuple[object, ...], ValueRef] = field(
        default_factory=dict
    )
    kernel_body_block: object | None = None
    dense_layout: ValueRef | None = None

    def type(self, value_type: str) -> Type:
        return self.type_converter.map_dtype(value_type)

    def default_address_layout(self) -> ValueRef:
        if self.dense_layout is None:
            self.dense_layout = self.builder.encoding.layout_dense(
                results=[ENCODING_LAYOUT_TYPE],
                name=self.reserve_name("layout"),
            )
        return self.dense_layout

    def buffer_view_type(self, buffer: object) -> ShapedType:
        view_type = self.type_converter.view_type(buffer)
        if view_type.encoding is not None:
            return view_type
        self.default_address_layout()
        return ShapedType(
            view_type.type_kind,
            view_type.element_type,
            view_type.dims,
            encoding=DynamicEncoding(),
        )

    def bind_buffer_view_layout(self, view: ValueRef) -> None:
        view_value = self.builder.module.values[view.id]
        view_type = view_value.type
        if isinstance(view_type, ShapedType) and isinstance(
            view_type.encoding, DynamicEncoding
        ):
            view_value.encoding_binding = self.default_address_layout().id

    def build_constant(self, value: Any, value_type: str, name: str) -> ValueRef:
        result_type = self.type_converter.map_dtype(
            value_type,
            index_like=value_type == "index",
        )
        if result_type in (INDEX, OFFSET):
            return self.builder.index.constant(
                value=int(value),
                results=[result_type],
                name=name,
            )
        return self.builder.scalar.constant(
            value=value,
            results=[result_type],
            name=name,
        )

    def map_value(
        self,
        source: object,
        ref: ValueRef,
        value_type: str | None = None,
    ) -> None:
        SourceImportSession.map_value(self, source, ref, value_type)
        semantic_key = _semantic_source_key(source)
        if semantic_key is not None:
            self.semantic_values[semantic_key] = ref
            self.semantic_value_types[semantic_key] = (
                value_type if value_type is not None else str(ref.type)
            )

    def mapped(self, source: object) -> ValueRef | None:
        mapped_value = SourceImportSession.mapped(self, source)
        if mapped_value is not None:
            return mapped_value
        semantic_key = _semantic_source_key(source)
        if semantic_key is None:
            return None
        return self.semantic_values.get(semantic_key)

    def mapped_value_type(self, source: object) -> str | None:
        mapped_type = SourceImportSession.mapped_value_type(self, source)
        if mapped_type is not None:
            return mapped_type
        semantic_key = _semantic_source_key(source)
        if semantic_key is None:
            return None
        return self.semantic_value_types.get(semantic_key)

    def map_buffer_data(self, source: object, ref: ValueRef) -> None:
        self.map_value(source, ref, str(ref.type))
        semantic_key = _semantic_buffer_data_key(source)
        if semantic_key is not None:
            self.buffer_data_values[semantic_key] = ref

    def mapped_buffer_data(self, source: object) -> ValueRef | None:
        mapped_value = self.mapped(source)
        if mapped_value is not None:
            return mapped_value
        semantic_key = _semantic_buffer_data_key(source)
        if semantic_key is None:
            return None
        return self.buffer_data_values.get(semantic_key)

    def mapped_index_value(self, source: object) -> ValueRef | None:
        mapped_value = self.index_values.get(self.source_key(source))
        if mapped_value is not None:
            return mapped_value
        semantic_key = _semantic_var_key(source)
        if semantic_key is None:
            return None
        return self.semantic_index_values.get(semantic_key)

    def map_index_value(self, source: object, ref: ValueRef) -> None:
        self.index_values[self.source_key(source)] = ref
        semantic_key = _semantic_var_key(source)
        if semantic_key is not None:
            self.semantic_index_values[semantic_key] = ref
        if ref.name:
            self.names.capture(ref.name)

    def can_emit_kernel_exit(self) -> bool:
        return (
            self.kernel_body_block is not None
            and self.builder.ir.insertion_block is self.kernel_body_block
        )

    def fork(self, *, preview_block: object | None = None) -> TileLangConversionContext:
        return TileLangConversionContext(
            builder=self.builder,
            diagnostics=self.diagnostics,
            preview_block=preview_block,
            value_map=dict(self.value_map),
            value_types=dict(self.value_types),
            constants=dict(self.constants),
            registry=self.registry,
            names=self.names,
            type_converter=self.type_converter,
            index_values=dict(self.index_values),
            semantic_values=dict(self.semantic_values),
            semantic_value_types=dict(self.semantic_value_types),
            buffer_data_values=dict(self.buffer_data_values),
            semantic_index_values=dict(self.semantic_index_values),
            kernel_body_block=self.kernel_body_block,
            dense_layout=self.dense_layout,
        )


def _semantic_source_key(source: object) -> tuple[object, ...] | None:
    var_key = _semantic_var_key(source)
    if var_key is not None:
        return var_key
    return _semantic_buffer_key(source)


def _semantic_var_key(source: object) -> tuple[object, ...] | None:
    if node_kind(source) not in ("Var", "SizeVar"):
        return None
    source_dtype = dtype(source)
    if source_dtype == "handle" or source_dtype.endswith("*"):
        return None
    return ("var", source_name(source, fallback=str(source)), source_dtype)


def _semantic_buffer_data_key(source: object) -> tuple[object, ...] | None:
    if node_kind(source) not in ("Var", "SizeVar"):
        return None
    source_dtype = dtype(source)
    if source_dtype != "handle" and not source_dtype.endswith("*"):
        return None
    return ("buffer_data", source_name(source, fallback=str(source)), source_dtype)


def _semantic_buffer_key(source: object) -> tuple[object, ...] | None:
    if node_kind(source) != "Buffer":
        return None
    data = getattr(source, "data", None)
    return (
        "buffer",
        source_name(source, fallback=str(source)),
        dtype(source),
        _buffer_scope(source),
        source_name(data, fallback=str(data)),
        _semantic_sequence(getattr(source, "shape", ())),
    )


def _semantic_sequence(value: object) -> tuple[object, ...]:
    if value is None or isinstance(value, str | bytes):
        return ()
    if isinstance(value, Iterable):
        return tuple(_semantic_value(item) for item in value)
    return (_semantic_value(value),)


def _semantic_value(value: object) -> object:
    if isinstance(value, str | bytes | int | float | bool):
        return value
    if node_kind(value) in ("Var", "SizeVar"):
        return _semantic_var_key(value)
    if hasattr(value, "value"):
        return ("imm", str(value.value), dtype(value))
    return (
        node_kind(value),
        source_name(value, fallback=str(value)),
        dtype(value),
        str(value),
    )


def _buffer_scope(buffer: object) -> str:
    scope = getattr(buffer, "scope", None)
    if callable(scope):
        return str(scope())
    if scope is not None:
        return str(scope)
    return ""
