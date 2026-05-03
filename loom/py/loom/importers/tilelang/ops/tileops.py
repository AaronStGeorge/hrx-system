# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TileLang tileop call converters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from loom.builder import ValueRef
from loom.importers.tilelang.context import TileLangConversionContext
from loom.importers.tilelang.converter import ExpressionOptions, TileLangConverter
from loom.importers.tilelang.nodes import node_kind, node_text
from loom.importers.tilelang.ops.topology import integer_value
from loom.ir import INDEX, ShapedType, Type


@dataclass(frozen=True, slots=True)
class TileRegion:
    """Decoded TileLang tile region over one Loom view."""

    view: ValueRef
    indices: tuple[ValueRef, ...]
    extents: tuple[ValueRef, ...]


def is_tileop_call(op_name: str) -> bool:
    """Returns true for calls owned by TileLang's tileop namespace."""

    return op_name.startswith("tl.tileop.")


def convert_tileop_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    options: ExpressionOptions,
    op_name: str,
) -> ValueRef | None:
    """Import a TileLang tileop call."""

    if op_name in _REGION_CALLS:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` must be consumed by a tile operation",
        )
        return None
    if op_name in _COPY_CALLS:
        return _convert_copy_call(expr, context, converter, options, op_name)
    if op_name in _FILL_CALLS:
        return _convert_fill_call(expr, context, converter, options, op_name)
    context.record_blocked(
        node_text(expr),
        f"TileLang tile operation `{op_name}` is not imported",
    )
    return None


def _convert_copy_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    options: ExpressionOptions,
    op_name: str,
) -> ValueRef | None:
    if not _require_effect(expr, context, options, op_name):
        return None
    args = _args(expr)
    if len(args) != 2:
        context.record_blocked(node_text(expr), f"call `{op_name}` expects 2 regions")
        return None
    source = _decode_region(args[0], expr, context, converter, expected_access=1)
    target = _decode_region(args[1], expr, context, converter, expected_access=2)
    if source is None or target is None:
        return None
    if not _same_region_shape(source, target):
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` source and destination extents differ",
        )
        return None
    element_type = _view_element_type(source.view, context)
    if element_type is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` source region is not a shaped view",
        )
        return None

    def emit_copy(
        indices: tuple[ValueRef, ...],
        body_context: TileLangConversionContext,
    ) -> None:
        source_indices = _region_indices(source, indices, body_context)
        target_indices = _region_indices(target, indices, body_context)
        value = body_context.builder.view.load(
            view=source.view,
            indices=list(source_indices),
            results=[element_type],
            name=body_context.fresh_name("copy"),
        )
        body_context.builder.view.store(
            value=value,
            view=target.view,
            indices=list(target_indices),
        )

    if not _emit_region_loops(source.extents, context, emit_copy):
        context.record_blocked(
            node_text(expr), f"call `{op_name}` region is not mapped"
        )
        return None
    context.record_converted(node_text(expr), "tl.tileop.copy")
    return None


def _convert_fill_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    options: ExpressionOptions,
    op_name: str,
) -> ValueRef | None:
    if not _require_effect(expr, context, options, op_name):
        return None
    args = _args(expr)
    if len(args) != 2:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` expects region and value",
        )
        return None
    target = _decode_region(args[0], expr, context, converter, expected_access=2)
    if target is None:
        return None
    element_type = _view_element_type(target.view, context)
    if element_type is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` target region is not a shaped view",
        )
        return None
    value = _convert_fill_value(args[1], element_type, context, converter)
    if value is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` fill value is not mapped",
        )
        return None

    def emit_fill(
        indices: tuple[ValueRef, ...],
        body_context: TileLangConversionContext,
    ) -> None:
        body_context.builder.view.store(
            value=value,
            view=target.view,
            indices=list(_region_indices(target, indices, body_context)),
        )

    if not _emit_region_loops(target.extents, context, emit_fill):
        context.record_blocked(
            node_text(expr), f"call `{op_name}` region is not mapped"
        )
        return None
    context.record_converted(node_text(expr), "tl.tileop.fill")
    return None


def _require_effect(
    expr: object,
    context: TileLangConversionContext,
    options: ExpressionOptions,
    op_name: str,
) -> bool:
    if options.effect:
        return True
    context.record_blocked(
        node_text(expr),
        f"call `{op_name}` is effect-only and must appear under tir.Evaluate",
    )
    return False


def _decode_region(
    expr: object,
    owner: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    expected_access: int,
) -> TileRegion | None:
    op_name = _call_op_name(expr)
    if op_name != "tl.tileop.region":
        context.record_blocked(
            node_text(owner),
            f"tile operation region must be tl.tileop.region, got `{op_name}`",
        )
        return None
    args = _args(expr)
    if len(args) < 2:
        context.record_blocked(
            node_text(owner),
            "tl.tileop.region expects buffer load, access mask, and extents",
        )
        return None
    buffer_load = args[0]
    if node_kind(buffer_load) != "BufferLoad":
        context.record_blocked(
            node_text(owner),
            "tl.tileop.region base is not a BufferLoad",
        )
        return None
    access_mask = integer_value(args[1])
    if access_mask is None:
        context.record_blocked(
            node_text(owner),
            "tl.tileop.region access mask is not static",
        )
        return None
    if access_mask & expected_access == 0:
        context.record_blocked(
            node_text(owner),
            f"tl.tileop.region access mask `{access_mask}` does not permit "
            "the operation",
        )
        return None
    buffer = getattr(buffer_load, "buffer", None)
    view = context.mapped(buffer)
    if view is None:
        context.record_blocked(
            node_text(owner),
            "tl.tileop.region buffer is not mapped",
        )
        return None
    indices = _convert_index_sequence(
        tuple(getattr(buffer_load, "indices", ())),
        owner,
        context,
        converter,
        what="region indices",
    )
    extents = _convert_index_sequence(
        args[2:],
        owner,
        context,
        converter,
        what="region extents",
    )
    if indices is None or extents is None:
        return None
    if len(indices) != len(extents):
        context.record_blocked(
            node_text(owner),
            "tl.tileop.region index and extent ranks differ",
        )
        return None
    return TileRegion(
        view=view,
        indices=tuple(indices),
        extents=tuple(extents),
    )


def _convert_index_sequence(
    values: Sequence[object],
    owner: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    what: str,
) -> list[ValueRef] | None:
    converted: list[ValueRef] = []
    for value in values:
        index = converter.convert_expr(value, context, index_like=True)
        if index is None:
            context.record_blocked(node_text(owner), f"tl.tileop {what} are not mapped")
            return None
        if str(index.type) != "index":
            context.record_blocked(
                node_text(owner),
                f"tl.tileop {what} must be scalar index values",
            )
            return None
        converted.append(index)
    return converted


def _same_region_shape(source: TileRegion, target: TileRegion) -> bool:
    if len(source.extents) != len(target.extents):
        return False
    return all(
        lhs.id == rhs.id
        for lhs, rhs in zip(source.extents, target.extents, strict=False)
    )


def _emit_region_loops(
    extents: tuple[ValueRef, ...],
    context: TileLangConversionContext,
    emit_body: Callable[[tuple[ValueRef, ...], TileLangConversionContext], None],
) -> bool:
    zero = context.ensure_constant("0", "index", "c0")
    one = context.ensure_constant("1", "index", "c1")

    def emit_at_depth(
        depth: int,
        indices: tuple[ValueRef, ...],
        loop_context: TileLangConversionContext,
    ) -> None:
        if depth == len(extents):
            emit_body(indices, loop_context)
            return
        loop_body = loop_context.builder.region(args=[(f"i{depth}", INDEX)])
        iv = ValueRef(loop_body.blocks[0].arg_ids[0], loop_context.builder.ir)
        child = loop_context.fork(preview_block=loop_body.blocks[0])
        with loop_context.builder.insertion_block(loop_body.blocks[0]):
            emit_at_depth(depth + 1, (*indices, iv), child)
            loop_context.builder.scf.yield_()
        loop_context.merge_child_records(child)
        loop_context.builder.scf.for_(
            lower_bound=zero,
            upper_bound=extents[depth],
            step=one,
            iter_args=[],
            results=[],
            body=loop_body,
        )

    emit_at_depth(0, (), context)
    return True


def _region_indices(
    region: TileRegion,
    loop_indices: tuple[ValueRef, ...],
    context: TileLangConversionContext,
) -> tuple[ValueRef, ...]:
    indices: list[ValueRef] = []
    for base, offset in zip(region.indices, loop_indices, strict=True):
        indices.append(_add_region_index(base, offset, context))
    return tuple(indices)


def _add_region_index(
    base: ValueRef,
    offset: ValueRef,
    context: TileLangConversionContext,
) -> ValueRef:
    zero = context.constants.get(("index", "0"))
    if zero is not None:
        if base.id == zero.id:
            return offset
        if offset.id == zero.id:
            return base
    return context.builder.index.add(
        lhs=base,
        rhs=offset,
        results=[INDEX],
        name=context.fresh_name("idx"),
    )


def _view_element_type(
    view: ValueRef,
    context: TileLangConversionContext,
) -> Type | None:
    view_type = context.builder.module.values[view.id].type
    if not isinstance(view_type, ShapedType):
        return None
    return view_type.element_type


def _convert_fill_value(
    source: object,
    element_type: Type,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> ValueRef | None:
    payload = getattr(source, "value", source)
    if isinstance(payload, int | float | bool):
        return context.builder.scalar.constant(
            value=payload,
            results=[element_type],
            name=context.reserve_name("const"),
        )
    return converter.convert_expr(source, context)


def _call_op_name(call: object) -> str | None:
    op = getattr(call, "op", None)
    if op is None:
        return None
    name = getattr(op, "name", None)
    if name:
        return str(name)
    get_name = getattr(op, "get_name", None)
    if get_name is None:
        return None
    return str(get_name())


def _args(call: object) -> tuple[object, ...]:
    return tuple(getattr(call, "args", ()))


_REGION_CALLS = {
    "tl.tileop.region",
}

_COPY_CALLS = {
    "tl.tileop.copy",
}

_FILL_CALLS = {
    "tl.tileop.fill",
}
