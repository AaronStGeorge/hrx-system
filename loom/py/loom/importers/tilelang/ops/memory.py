# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Buffer load/store TileLang/TIR node converters."""

from __future__ import annotations

from loom.builder import ValueRef
from loom.importers.core import sanitize_identifier
from loom.importers.tilelang.context import TileLangConversionContext
from loom.importers.tilelang.converter import (
    ExpressionOptions,
    TileLangConverter,
    TileLangConverterRegistry,
)
from loom.importers.tilelang.nodes import dtype, node_kind, node_text, source_name
from loom.importers.tilelang.ops.topology import integer_value
from loom.importers.tilelang.ops.vector import vector_lanes
from loom.ir import (
    BUFFER_TYPE,
    INDEX,
    DynamicDim,
    ShapedType,
    StaticDim,
    Type,
    TypeKind,
)


def register(registry: TileLangConverterRegistry) -> None:
    registry.register_statement("BufferStore", convert_buffer_store)
    registry.register_expression("BufferLoad", convert_buffer_load)


def map_alloc_buffer(
    buffer: object,
    context: TileLangConversionContext,
) -> bool:
    """Import a TIR block allocation as a Loom scratch buffer root."""

    byte_length = context.type_converter.buffer_byte_length(buffer)
    if byte_length is None:
        context.record_blocked(
            node_text(buffer),
            "allocated buffer byte length is not statically known",
        )
        return False
    memory_space = _buffer_memory_space(buffer)
    if memory_space is None:
        context.record_blocked(
            node_text(buffer),
            f"allocated buffer scope `{_buffer_scope(buffer)}` is not mapped",
        )
        return False
    buffer_name = sanitize_identifier(source_name(buffer, fallback="scratch"))
    root = context.builder.buffer.alloca(
        byte_length=context.ensure_constant(
            str(byte_length),
            "offset",
            f"{buffer_name}_bytes",
        ),
        base_alignment=context.type_converter.buffer_base_alignment(buffer),
        memory_space=memory_space,
        results=[BUFFER_TYPE],
        name=context.fresh_name(f"{buffer_name}_buffer"),
    )
    view_type = context.buffer_view_type(buffer)
    view = context.builder.buffer.view(
        buffer=root,
        byte_offset=context.ensure_constant("0", "offset", "c0_bytes"),
        results=[view_type],
        name=context.fresh_name(buffer_name),
    )
    context.bind_buffer_view_layout(view)
    context.map_value(buffer, view, str(view.type))
    data = getattr(buffer, "data", None)
    if data is not None:
        context.map_buffer_data(data, view, buffer=buffer)
    context.record_converted(
        node_text(buffer),
        (
            f"{context.ssa(root)} = buffer.alloca",
            f"{context.ssa(view)} = buffer.view",
        ),
    )
    return True


def convert_buffer_store(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Import a TIR buffer store as view.store."""

    source_indices = tuple(getattr(stmt, "indices", ()))
    buffer = getattr(stmt, "buffer", None)
    view, source_indices = _resolve_buffer_view(
        buffer,
        source_indices,
        context,
        converter,
        diagnostic_owner=stmt,
    )
    if view is None:
        return
    value = converter.convert_expr(getattr(stmt, "value", None), context)
    if value is None:
        context.record_blocked(node_text(stmt), "buffer store operands are not mapped")
        return
    has_ramp_index = _has_ramp_index(stmt)
    if has_ramp_index and not is_vector_type(value.type):
        context.record_blocked(
            node_text(stmt), "vector store value is not vector-typed"
        )
        return
    if is_vector_type(value.type) or has_ramp_index:
        vector_indices = _vector_memory_indices(
            source_indices,
            context,
            converter,
        )
        if vector_indices is None:
            context.record_blocked(
                node_text(stmt), "vector store indices are not mapped"
            )
            return
        context.builder.vector.store(
            value=value,
            view=view,
            indices=vector_indices,
        )
        context.record_converted(node_text(stmt), "vector.store")
        return
    indices = [_convert_index(index, context, converter) for index in source_indices]
    if any(index is None for index in indices):
        context.record_blocked(node_text(stmt), "buffer store operands are not mapped")
        return
    value = _coerce_store_value(value, view, stmt, context)
    if value is None:
        return
    mapped_indices: list[int | ValueRef] = [
        index for index in indices if index is not None
    ]
    context.builder.view.store(value=value, view=view, indices=mapped_indices)
    context.record_converted(node_text(stmt), "view.store")


def convert_buffer_load(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    options: ExpressionOptions,
) -> ValueRef | None:
    """Import a TIR buffer load as view.load."""

    source_indices = tuple(getattr(expr, "indices", ()))
    buffer = getattr(expr, "buffer", None)
    view, source_indices = _resolve_buffer_view(
        buffer,
        source_indices,
        context,
        converter,
        diagnostic_owner=expr,
    )
    if view is None:
        return None
    result_type = context.type_converter.map_dtype(dtype(expr))
    if is_vector_type(result_type) or _has_ramp_index(expr):
        vector_indices = _vector_memory_indices(source_indices, context, converter)
        if vector_indices is None:
            context.record_blocked(
                node_text(expr), "vector load indices are not mapped"
            )
            return None
        if not is_vector_type(result_type):
            lanes = _single_ramp_lanes(source_indices)
            if lanes is None:
                context.record_blocked(
                    node_text(expr), "vector load lanes are not mapped"
                )
                return None
            result_type = context.type_converter.vector_type(dtype(expr), lanes)
        result = context.builder.vector.load(
            view=view,
            indices=vector_indices,
            results=[result_type],
            name=context.fresh_name("load"),
        )
        context.map_value(expr, result, str(result_type))
        context.record_converted(
            node_text(expr), f"{context.ssa(result)} = vector.load"
        )
        return result
    indices = [_convert_index(index, context, converter) for index in source_indices]
    if any(index is None for index in indices):
        context.record_blocked(node_text(expr), "buffer load indices are not mapped")
        return None
    result = context.builder.view.load(
        view=view,
        indices=[index for index in indices if index is not None],
        results=[result_type],
        name=context.fresh_name("load"),
    )
    context.map_value(expr, result, str(result_type))
    context.record_converted(node_text(expr), f"{context.ssa(result)} = view.load")
    if options.index_like:
        index_result = _coerce_index_like_load(result, expr, context)
        if index_result is None:
            return None
        return index_result
    return result


def _resolve_buffer_view(
    buffer: object,
    source_indices: tuple[object, ...],
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    diagnostic_owner: object,
) -> tuple[ValueRef | None, tuple[object | ValueRef, ...]]:
    view = context.mapped(buffer)
    if view is not None:
        return view, source_indices
    data = getattr(buffer, "data", None)
    view = context.mapped_buffer_data(data)
    if view is None:
        context.record_blocked(
            node_text(diagnostic_owner),
            "buffer access target is not mapped",
        )
        return None, source_indices
    view_type = _view_type(view, context)
    if isinstance(view_type, ShapedType) and len(view_type.dims) <= 1:
        return view, source_indices
    remapped_indices = remap_flattened_indices(
        view,
        source_indices,
        context,
        converter,
        diagnostic_owner=diagnostic_owner,
    )
    if remapped_indices is None:
        return None, source_indices
    return view, tuple(remapped_indices)


def _convert_index(
    index: object | ValueRef,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> ValueRef | None:
    if isinstance(index, ValueRef):
        return index
    return converter.convert_expr(index, context, index_like=True)


def _coerce_index_like_load(
    value: ValueRef,
    source: object,
    context: TileLangConversionContext,
) -> ValueRef | None:
    if str(value.type) in ("index", "offset"):
        context.map_index_value(source, value)
        return value
    if _integer_bit_width(str(value.type)) is None:
        context.record_blocked(
            node_text(source),
            f"buffer load value conversion {value.type} to index is not imported",
        )
        return None
    result = context.builder.index.cast(
        input=value,
        results=[INDEX],
        name=context.fresh_name("load_idx"),
    )
    context.map_index_value(source, result)
    context.record_converted(node_text(source), f"{context.ssa(result)} = index.cast")
    return result


def _coerce_store_value(
    value: ValueRef,
    view: ValueRef,
    owner: object,
    context: TileLangConversionContext,
) -> ValueRef | None:
    view_type = _view_type(view, context)
    if not isinstance(view_type, ShapedType):
        context.record_blocked(node_text(owner), "buffer store target is not shaped")
        return None
    return _cast_store_value(value, view_type.element_type, owner, context)


def _cast_store_value(
    value: ValueRef,
    target_type: Type,
    owner: object,
    context: TileLangConversionContext,
) -> ValueRef | None:
    source_type = value.type
    if str(source_type) == str(target_type):
        return value
    source_text = str(source_type)
    target_text = str(target_type)
    if (
        source_text in ("index", "offset")
        and _integer_bit_width(target_text) is not None
    ):
        return context.builder.index.cast(
            input=value,
            results=[target_type],
            name=context.fresh_name("store_cast"),
        )
    source_width = _integer_bit_width(source_text)
    target_width = _integer_bit_width(target_text)
    if source_width is not None and target_width is not None:
        if source_width < target_width:
            return context.builder.scalar.extsi(
                input=value,
                results=[target_type],
                name=context.fresh_name("store_ext"),
            )
        if source_width > target_width:
            return context.builder.scalar.trunci(
                input=value,
                results=[target_type],
                name=context.fresh_name("store_trunc"),
            )
    context.record_blocked(
        node_text(owner),
        f"buffer store value conversion {source_type} to {target_type} is not imported",
    )
    return None


def _integer_bit_width(type_text: str) -> int | None:
    if not type_text.startswith("i") or not type_text[1:].isdecimal():
        return None
    return int(type_text[1:])


def remap_flattened_indices(
    view: ValueRef,
    source_indices: tuple[object, ...],
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    diagnostic_owner: object,
) -> list[ValueRef] | None:
    """Map a TileLang flattened alias index back onto the source view rank."""

    view_type = _view_type(view, context)
    if not isinstance(view_type, ShapedType) or len(view_type.dims) <= 1:
        context.record_blocked(
            node_text(diagnostic_owner),
            "buffer alias target does not have a remappable shaped view type",
        )
        return None
    if len(source_indices) != 1:
        context.record_blocked(
            node_text(diagnostic_owner),
            "buffer alias access has non-flat indices",
        )
        return None
    flat_index = converter.convert_expr(source_indices[0], context, index_like=True)
    if flat_index is None:
        context.record_blocked(
            node_text(diagnostic_owner),
            "buffer alias flat index is not mapped",
        )
        return None
    remapped: list[ValueRef | None] = [None] * len(view_type.dims)
    running = flat_index
    for position in range(len(view_type.dims) - 1, 0, -1):
        extent = _view_dim_extent(view, position, context)
        if extent is None:
            context.record_blocked(
                node_text(diagnostic_owner),
                "buffer alias target dimension is not bound",
            )
            return None
        remapped[position] = context.builder.index.rem(
            lhs=running,
            rhs=extent,
            results=[running.type],
            name=context.fresh_name("rem"),
        )
        running = context.builder.index.div(
            lhs=running,
            rhs=extent,
            results=[running.type],
            name=context.fresh_name("div"),
        )
    remapped[0] = running
    return [index for index in remapped if index is not None]


def _view_dim_extent(
    view: ValueRef,
    position: int,
    context: TileLangConversionContext,
) -> ValueRef | None:
    view_value = context.builder.module.values[view.id]
    view_type = _view_type(view, context)
    if not isinstance(view_type, ShapedType):
        return None
    dim = view_type.dims[position]
    if isinstance(dim, StaticDim):
        return context.ensure_constant(str(dim.size), "index", f"c{dim.size}")
    if isinstance(dim, DynamicDim):
        value_id = view_value.dim_bindings.get(position)
        if value_id is None:
            return None
        return ValueRef(value_id, context.builder.ir)
    return None


def _view_type(view: ValueRef, context: TileLangConversionContext) -> object:
    return context.builder.module.values[view.id].type


def is_vector_type(value_type: object) -> bool:
    return (
        isinstance(value_type, ShapedType) and value_type.type_kind == TypeKind.VECTOR
    )


def _has_ramp_index(expr: object) -> bool:
    return any(
        node_kind(index) == "Ramp" for index in tuple(getattr(expr, "indices", ()))
    )


def _vector_memory_indices(
    source_indices: tuple[object, ...],
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> list[int | ValueRef] | None:
    mapped_indices: list[int | ValueRef] = []
    ramp_count = 0
    for source_index in source_indices:
        if node_kind(source_index) == "Ramp":
            ramp_count += 1
            if ramp_count > 1:
                return None
            stride = integer_value(getattr(source_index, "stride", None))
            if stride != 1:
                return None
            base = converter.convert_expr(
                getattr(source_index, "base", None),
                context,
                index_like=True,
            )
            if base is None:
                return None
            mapped_indices.append(base)
            continue
        mapped = converter.convert_expr(source_index, context, index_like=True)
        if mapped is None:
            return None
        mapped_indices.append(mapped)
    return mapped_indices


def _single_ramp_lanes(source_indices: tuple[object, ...]) -> int | None:
    lanes: int | None = None
    for source_index in source_indices:
        if node_kind(source_index) != "Ramp":
            continue
        if lanes is not None:
            return None
        lanes = vector_lanes(source_index)
    return lanes


def _buffer_scope(buffer: object) -> str:
    scope = getattr(buffer, "scope", None)
    if callable(scope):
        return str(scope())
    if scope is not None:
        return str(scope)
    return ""


def _buffer_memory_space(buffer: object) -> str | None:
    scope = _buffer_scope(buffer)
    if scope in ("", "local", "local.dyn", "local.fragment", "local.var"):
        return "private"
    if scope in ("shared", "shared.dyn"):
        return "workgroup"
    return None
