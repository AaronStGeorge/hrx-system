# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TileLang tileop call converters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

from loom.builder import ValueRef
from loom.importers.tilelang.context import TileLangConversionContext
from loom.importers.tilelang.converter import ExpressionOptions, TileLangConverter
from loom.importers.tilelang.nodes import dtype, mapping_items, node_kind, node_text
from loom.importers.tilelang.ops.topology import integer_value
from loom.ir import INDEX, ShapedType, StaticDim, Type, TypeKind


@dataclass(frozen=True, slots=True)
class TileRegion:
    """Decoded TileLang tile region over one Loom view."""

    view: ValueRef
    indices: tuple[ValueRef, ...]
    extents: tuple[ValueRef, ...]


@dataclass(frozen=True, slots=True)
class TileBufferAccess:
    """Decoded TileLang buffer access over one Loom view."""

    view: ValueRef
    indices: tuple[ValueRef, ...]


@dataclass(frozen=True, slots=True)
class TileReduceSpec:
    """Decoded TileLang reduction semantics over one element stream."""

    source_kind: str
    combiner: str
    absolute: bool = False


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
    if op_name in _REDUCE_CALLS:
        return _convert_reduce_call(expr, context, converter, options, op_name)
    if op_name in _FINALIZE_REDUCER_CALLS:
        return _convert_finalize_reducer_call(
            expr, context, converter, options, op_name
        )
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
    if not _validate_copy_annotations(expr, context):
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


def _validate_copy_annotations(
    expr: object,
    context: TileLangConversionContext,
) -> bool:
    annotations = _call_annotations(expr)
    unknown_annotations = sorted(set(annotations) - _COPY_ANNOTATIONS)
    if unknown_annotations:
        context.record_blocked(
            node_text(expr),
            (
                "tl.tileop.copy annotations are not imported: "
                + ", ".join(unknown_annotations)
            ),
        )
        return False
    if "disable_tma" in annotations:
        disable_tma = _static_bool(annotations["disable_tma"])
        if disable_tma is None:
            context.record_blocked(
                node_text(expr),
                "tl.tileop.copy annotation `disable_tma` is not static",
            )
            return False
    for annotation in sorted(set(annotations) - {"disable_tma"}):
        context.record_blocked(
            node_text(expr),
            f"copy annotation `{annotation}` needs scheduling import",
        )
        return False
    return True


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


def _convert_reduce_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    options: ExpressionOptions,
    op_name: str,
) -> ValueRef | None:
    if not _require_effect(expr, context, options, op_name):
        return None
    args = _args(expr)
    if len(args) != 5:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` expects source, destination, kind, dim, and clear",
        )
        return None
    dim = integer_value(args[3])
    if dim is None:
        context.record_blocked(node_text(expr), f"call `{op_name}` dim is not static")
        return None
    clear = integer_value(args[4])
    if clear is None:
        context.record_blocked(node_text(expr), f"call `{op_name}` clear is not static")
        return None
    source_axis = _single_ramp_axis(args[0])
    if source_axis is None:
        return _convert_region_reduce_call(
            expr,
            context,
            converter,
            op_name,
            dim=dim,
            clear=bool(clear),
            kind_expr=args[2],
        )
    if source_axis != dim:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` dim {dim} does not match source ramp axis {source_axis}",
        )
        return None
    source = converter.convert_expr(args[0], context)
    if source is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` source vector is not mapped",
        )
        return None
    if not _is_rank_one_vector_type(source.type):
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` source must map to a rank-1 vector",
        )
        return None
    source_type = cast(ShapedType, source.type)
    element_type = source_type.element_type
    target = _decode_buffer_access(args[1], expr, context, converter)
    if target is None:
        return None
    target_element_type = _view_element_type(target.view, context)
    if target_element_type is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` destination is not a shaped view",
        )
        return None
    if str(target_element_type) != str(element_type):
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` source and destination element types differ",
        )
        return None
    reduce_spec = _reduce_spec(args[2], args[0], element_type, expr, context)
    if reduce_spec is None:
        return None
    init = _reduce_init(
        bool(clear),
        reduce_spec,
        element_type,
        target,
        expr,
        context,
    )
    if init is None:
        return None
    source = _build_vector_reduce_input(source, reduce_spec, element_type, context)
    result = context.builder.vector.reduce(
        kind=reduce_spec.combiner,
        input=source,
        init=init,
        results=[element_type],
        name=context.fresh_name("reduce"),
    )
    context.builder.view.store(
        value=result,
        view=target.view,
        indices=list(target.indices),
    )
    context.record_converted(
        node_text(expr),
        f"tl.tileop.reduce<{reduce_spec.source_kind}>",
    )
    return None


def _convert_region_reduce_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    op_name: str,
    *,
    dim: int,
    clear: bool,
    kind_expr: object,
) -> ValueRef | None:
    source = _decode_region_arg(
        args=_args(expr), position=0, expr=expr, context=context, converter=converter
    )
    target = _decode_region_arg(
        args=_args(expr), position=1, expr=expr, context=context, converter=converter
    )
    if source is None or target is None:
        return None
    if dim < 0 or dim >= len(source.extents):
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` dim {dim} is outside source rank {len(source.extents)}",
        )
        return None
    element_type = _view_element_type(source.view, context)
    if element_type is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` source region is not a shaped view",
        )
        return None
    target_element_type = _view_element_type(target.view, context)
    if target_element_type is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` destination region is not a shaped view",
        )
        return None
    if str(target_element_type) != str(element_type):
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` source and destination element types differ",
        )
        return None
    reduce_spec = _reduce_spec(kind_expr, _args(expr)[0], element_type, expr, context)
    if reduce_spec is None:
        return None
    output_extents = tuple(
        extent for axis, extent in enumerate(source.extents) if axis != dim
    )

    def emit_reduce(
        output_indices: tuple[ValueRef, ...],
        body_context: TileLangConversionContext,
    ) -> None:
        target_indices = _reduction_target_indices(
            target,
            len(source.extents),
            dim,
            output_indices,
            body_context,
        )
        if target_indices is None:
            body_context.record_blocked(
                node_text(expr),
                f"call `{op_name}` destination rank does not match reduced shape",
            )
            return
        init = _reduce_init(
            clear,
            reduce_spec,
            element_type,
            TileBufferAccess(target.view, target_indices),
            expr,
            body_context,
        )
        if init is None:
            return
        result = _emit_scalar_reduce_loop(
            source,
            dim,
            output_indices,
            init,
            reduce_spec,
            element_type,
            body_context,
        )
        body_context.builder.view.store(
            value=result,
            view=target.view,
            indices=list(target_indices),
        )

    if not _emit_region_loops(output_extents, context, emit_reduce):
        context.record_blocked(
            node_text(expr), f"call `{op_name}` output region is not mapped"
        )
        return None
    context.record_converted(
        node_text(expr),
        f"tl.tileop.reduce<{reduce_spec.source_kind}>",
    )
    return None


def _convert_finalize_reducer_call(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    options: ExpressionOptions,
    op_name: str,
) -> ValueRef | None:
    if not _require_effect(expr, context, options, op_name):
        return None
    args = _args(expr)
    if len(args) not in (1, 2):
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` expects reducer region and optional op code",
        )
        return None
    data_source = _reducer_data_source(args[0], expr, context)
    if data_source is None:
        return None
    reducer_info = context.reducer_info(data_source)
    if reducer_info is None:
        context.record_blocked(
            node_text(expr),
            "tl.tileop.finalize_reducer reducer_info metadata is missing",
        )
        return None
    if len(args) == 2:
        op_code = integer_value(args[1])
        expected_op_code = _REDUCER_OPERATION_CODES.get(reducer_info.operation)
        if op_code is None or op_code != expected_op_code:
            context.record_blocked(
                node_text(expr),
                "tl.tileop.finalize_reducer op code does not match reducer_info",
            )
            return None
    if reducer_info.replication == "none":
        context.record_converted(
            node_text(expr),
            f"tl.tileop.finalize_reducer<{reducer_info.operation}, none> normalized",
        )
        return None
    if reducer_info.replication == "all":
        context.record_blocked(
            node_text(expr),
            "replication `all` requires cross-thread allreduce import",
        )
        return None
    context.record_blocked(
        node_text(expr),
        (
            "tl.tileop.finalize_reducer replication "
            f"`{reducer_info.replication}` is not imported"
        ),
    )
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


def _decode_region_arg(
    *,
    args: tuple[object, ...],
    position: int,
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> TileRegion | None:
    value = args[position]
    if _call_op_name(value) == "tl.tileop.region":
        return _decode_region(value, expr, context, converter, expected_access=3)
    if node_kind(value) == "BufferRegion":
        return _decode_buffer_region(value, expr, context, converter)
    if node_kind(value) == "BufferLoad":
        return _decode_buffer_load_region(value, expr, context, converter)
    context.record_blocked(
        node_text(expr),
        f"tile operation region argument must be region-like, got `{node_kind(value)}`",
    )
    return None


def _decode_buffer_region(
    expr: object,
    owner: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> TileRegion | None:
    buffer = getattr(expr, "buffer", None)
    view = context.mapped(buffer)
    if view is None:
        data = getattr(buffer, "data", None)
        view = context.mapped_buffer_data(data)
    if view is None:
        context.record_blocked(
            node_text(owner),
            "tile operation buffer region target is not mapped",
        )
        return None
    ranges = tuple(getattr(expr, "region", ()))
    indices = _convert_index_sequence(
        tuple(getattr(item, "min", None) for item in ranges),
        owner,
        context,
        converter,
        what="region indices",
    )
    extents = _convert_index_sequence(
        tuple(getattr(item, "extent", None) for item in ranges),
        owner,
        context,
        converter,
        what="region extents",
    )
    if indices is None or extents is None:
        return None
    return TileRegion(view=view, indices=tuple(indices), extents=tuple(extents))


def _decode_buffer_load_region(
    expr: object,
    owner: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> TileRegion | None:
    buffer = getattr(expr, "buffer", None)
    view = context.mapped(buffer)
    if view is None:
        data = getattr(buffer, "data", None)
        view = context.mapped_buffer_data(data)
    if view is None:
        context.record_blocked(
            node_text(owner),
            "tile operation buffer load region target is not mapped",
        )
        return None
    one = context.ensure_constant("1", "index", "c1")
    indices: list[ValueRef] = []
    extents: list[ValueRef] = []
    for index in tuple(getattr(expr, "indices", ())):
        if node_kind(index) == "Ramp":
            stride = integer_value(getattr(index, "stride", None))
            lanes = integer_value(getattr(index, "lanes", None))
            if stride != 1 or lanes is None:
                context.record_blocked(
                    node_text(owner),
                    "tile operation ramp region requires unit stride and static lanes",
                )
                return None
            base = converter.convert_expr(
                getattr(index, "base", None),
                context,
                index_like=True,
            )
            if base is None:
                context.record_blocked(
                    node_text(owner),
                    "tile operation ramp region base is not mapped",
                )
                return None
            indices.append(base)
            extents.append(context.ensure_constant(str(lanes), "index", f"c{lanes}"))
            continue
        mapped = converter.convert_expr(index, context, index_like=True)
        if mapped is None:
            context.record_blocked(
                node_text(owner),
                "tile operation buffer load region index is not mapped",
            )
            return None
        indices.append(mapped)
        extents.append(one)
    return TileRegion(
        view=view,
        indices=tuple(indices),
        extents=tuple(extents),
    )


def _decode_buffer_access(
    expr: object,
    owner: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> TileBufferAccess | None:
    if node_kind(expr) != "BufferLoad":
        context.record_blocked(
            node_text(owner),
            f"tile operation buffer access must be BufferLoad, got `{node_kind(expr)}`",
        )
        return None
    buffer = getattr(expr, "buffer", None)
    view = context.mapped(buffer)
    if view is None:
        data = getattr(buffer, "data", None)
        view = context.mapped_buffer_data(data)
    if view is None:
        context.record_blocked(
            node_text(owner),
            "tile operation buffer access target is not mapped",
        )
        return None
    indices = _convert_index_sequence(
        tuple(getattr(expr, "indices", ())),
        owner,
        context,
        converter,
        what="buffer indices",
    )
    if indices is None:
        return None
    return TileBufferAccess(view=view, indices=tuple(indices))


def _reducer_data_source(
    expr: object,
    owner: object,
    context: TileLangConversionContext,
) -> object | None:
    region_like = expr
    if _call_op_name(region_like) == "tl.tileop.region":
        args = _args(region_like)
        if not args:
            context.record_blocked(
                node_text(owner),
                "tl.tileop.finalize_reducer region has no base",
            )
            return None
        region_like = args[0]
    if node_kind(region_like) == "BufferLoad":
        buffer = getattr(region_like, "buffer", None)
        data = getattr(buffer, "data", None)
        if data is not None:
            return cast(object, data)
    if node_kind(region_like) == "BufferRegion":
        buffer = getattr(region_like, "buffer", None)
        data = getattr(buffer, "data", None)
        if data is not None:
            return cast(object, data)
    context.record_blocked(
        node_text(owner),
        (
            "tl.tileop.finalize_reducer target must be a reducer BufferLoad, "
            f"BufferRegion, or tl.tileop.region, got `{node_kind(expr)}`"
        ),
    )
    return None


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


def _reduction_target_indices(
    target: TileRegion,
    source_rank: int,
    dim: int,
    output_indices: tuple[ValueRef, ...],
    context: TileLangConversionContext,
) -> tuple[ValueRef, ...] | None:
    if len(target.indices) == len(output_indices):
        return _region_indices(target, output_indices, context)
    if len(target.indices) == source_rank:
        expanded: list[ValueRef] = []
        output_axis = 0
        zero = context.ensure_constant("0", "index", "c0")
        for axis in range(source_rank):
            if axis == dim:
                expanded.append(zero)
                continue
            expanded.append(output_indices[output_axis])
            output_axis += 1
        return _region_indices(target, tuple(expanded), context)
    if not output_indices and len(target.indices) == 1:
        return target.indices
    return None


def _emit_scalar_reduce_loop(
    source: TileRegion,
    dim: int,
    output_indices: tuple[ValueRef, ...],
    init: ValueRef,
    reduce_spec: TileReduceSpec,
    element_type: Type,
    context: TileLangConversionContext,
) -> ValueRef:
    zero = context.ensure_constant("0", "index", "c0")
    one = context.ensure_constant("1", "index", "c1")
    loop_body = context.builder.region(args=[("r", INDEX), ("acc", element_type)])
    reduce_index = ValueRef(loop_body.blocks[0].arg_ids[0], context.builder.ir)
    accumulator = ValueRef(loop_body.blocks[0].arg_ids[1], context.builder.ir)
    child = context.fork(preview_block=loop_body.blocks[0])
    with context.builder.insertion_block(loop_body.blocks[0]):
        source_indices = _reduction_source_indices(
            source,
            dim,
            output_indices,
            reduce_index,
            child,
        )
        value = child.builder.view.load(
            view=source.view,
            indices=list(source_indices),
            results=[element_type],
            name=child.fresh_name("reduce_value"),
        )
        value = _build_scalar_reduce_input(value, reduce_spec, element_type, child)
        combined = _build_scalar_combiner(
            reduce_spec.combiner,
            accumulator,
            value,
            element_type,
            child,
        )
        child.builder.scf.yield_(values=[combined])
    context.merge_child_records(child)
    loop_results = context.builder.scf.for_(
        lower_bound=zero,
        upper_bound=source.extents[dim],
        step=one,
        iter_args=[init],
        results=[element_type],
        body=loop_body,
        names=[context.fresh_name("reduce")],
    )
    return loop_results[0]


def _reduction_source_indices(
    source: TileRegion,
    dim: int,
    output_indices: tuple[ValueRef, ...],
    reduce_index: ValueRef,
    context: TileLangConversionContext,
) -> tuple[ValueRef, ...]:
    source_offsets: list[ValueRef] = []
    output_axis = 0
    for axis in range(len(source.indices)):
        if axis == dim:
            source_offsets.append(reduce_index)
            continue
        source_offsets.append(output_indices[output_axis])
        output_axis += 1
    return _region_indices(source, tuple(source_offsets), context)


def _build_vector_reduce_input(
    source: ValueRef,
    reduce_spec: TileReduceSpec,
    element_type: Type,
    context: TileLangConversionContext,
) -> ValueRef:
    if not reduce_spec.absolute:
        return source
    vector_type = context.builder.module.values[source.id].type
    builder_name = "absf" if _is_float_type(element_type) else "absi"
    builder = getattr(context.builder.vector, builder_name)
    return cast(
        ValueRef,
        builder(
            input=source,
            results=[vector_type],
            name=context.fresh_name("abs"),
        ),
    )


def _build_scalar_reduce_input(
    source: ValueRef,
    reduce_spec: TileReduceSpec,
    element_type: Type,
    context: TileLangConversionContext,
) -> ValueRef:
    if not reduce_spec.absolute:
        return source
    builder_name = "absf" if _is_float_type(element_type) else "absi"
    builder = getattr(context.builder.scalar, builder_name)
    return cast(
        ValueRef,
        builder(
            input=source,
            results=[element_type],
            name=context.fresh_name("abs"),
        ),
    )


def _build_scalar_combiner(
    kind: str,
    lhs: ValueRef,
    rhs: ValueRef,
    element_type: Type,
    context: TileLangConversionContext,
) -> ValueRef:
    builder = getattr(context.builder.scalar, kind)
    return cast(
        ValueRef,
        builder(
            lhs=lhs,
            rhs=rhs,
            results=[element_type],
            name=context.fresh_name(kind),
        ),
    )


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


def _is_rank_one_vector_type(value_type: Type) -> bool:
    return (
        isinstance(value_type, ShapedType)
        and value_type.type_kind == TypeKind.VECTOR
        and len(value_type.dims) == 1
        and isinstance(value_type.dims[0], StaticDim)
    )


def _single_ramp_axis(expr: object) -> int | None:
    if node_kind(expr) != "BufferLoad":
        return None
    ramp_axis: int | None = None
    for axis, index in enumerate(tuple(getattr(expr, "indices", ()))):
        if node_kind(index) != "Ramp":
            continue
        if ramp_axis is not None:
            return None
        ramp_axis = axis
    return ramp_axis


def _reduce_spec(
    kind_expr: object,
    source: object,
    element_type: Type,
    owner: object,
    context: TileLangConversionContext,
) -> TileReduceSpec | None:
    reduce_type = getattr(kind_expr, "value", None)
    if reduce_type is None:
        context.record_blocked(node_text(owner), "tl.tileop.reduce kind is not static")
        return None
    reduce_type = str(reduce_type)
    annotations = _call_annotations(owner)
    unknown_annotations = sorted(set(annotations) - {"nan_propagate"})
    if unknown_annotations:
        context.record_blocked(
            node_text(owner),
            (
                "tl.tileop.reduce annotations are not imported: "
                + ", ".join(unknown_annotations)
            ),
        )
        return None
    nan_propagate = bool(annotations.get("nan_propagate", False))
    element_text = str(element_type)
    source_dtype = _scalar_dtype_text(dtype(source))
    is_float = element_text in ("f16", "bf16", "f32", "f64")
    is_unsigned = source_dtype.startswith(("uint", "u"))
    if reduce_type == "sum":
        return TileReduceSpec(reduce_type, "addf" if is_float else "addi")
    if reduce_type == "abssum":
        return TileReduceSpec(
            reduce_type,
            "addf" if is_float else "addi",
            absolute=True,
        )
    if reduce_type == "max":
        if is_float:
            return TileReduceSpec(
                reduce_type,
                "maximumf" if nan_propagate else "maxnumf",
            )
        return TileReduceSpec(reduce_type, "maxui" if is_unsigned else "maxsi")
    if reduce_type == "absmax":
        if is_float:
            return TileReduceSpec(
                reduce_type,
                "maximumf" if nan_propagate else "maxnumf",
                absolute=True,
            )
        return TileReduceSpec(
            reduce_type,
            "maxui" if is_unsigned else "maxsi",
            absolute=True,
        )
    if reduce_type == "min":
        if is_float:
            return TileReduceSpec(
                reduce_type,
                "minimumf" if nan_propagate else "minnumf",
            )
        return TileReduceSpec(reduce_type, "minui" if is_unsigned else "minsi")
    if reduce_type == "bitand" and not is_float:
        return TileReduceSpec(reduce_type, "andi")
    if reduce_type == "bitor" and not is_float:
        return TileReduceSpec(reduce_type, "ori")
    if reduce_type == "bitxor" and not is_float:
        return TileReduceSpec(reduce_type, "xori")
    context.record_blocked(
        node_text(owner),
        f"tl.tileop.reduce kind `{reduce_type}` is not imported",
    )
    return None


def _reduce_init(
    clear: bool,
    reduce_spec: TileReduceSpec,
    element_type: Type,
    target: TileBufferAccess,
    owner: object,
    context: TileLangConversionContext,
) -> ValueRef | None:
    if not clear:
        return context.builder.view.load(
            view=target.view,
            indices=list(target.indices),
            results=[element_type],
            name=context.fresh_name("reduce_init"),
        )
    identity = _reduce_identity(reduce_spec, element_type)
    if identity is None:
        context.record_blocked(
            node_text(owner),
            (
                f"tl.tileop.reduce<{reduce_spec.source_kind}> identity "
                f"is not imported for {element_type}"
            ),
        )
        return None
    return context.builder.scalar.constant(
        value=identity,
        results=[element_type],
        name=context.reserve_name("identity"),
    )


def _reduce_identity(
    reduce_spec: TileReduceSpec,
    element_type: Type,
) -> int | float | None:
    if reduce_spec.source_kind == "absmax":
        return 0
    kind = reduce_spec.combiner
    element_text = str(element_type)
    bit_width = _integer_bit_width(element_text)
    if kind in ("addf", "addi", "ori", "xori"):
        return 0
    if kind in ("mulf", "muli"):
        return 1
    if kind == "andi":
        return -1
    if kind in ("maxnumf", "maximumf"):
        return float("-inf")
    if kind in ("minnumf", "minimumf"):
        return float("inf")
    if kind == "maxui":
        return 0
    if kind == "minui":
        return -1
    if bit_width is None:
        return None
    if kind == "maxsi":
        return -(1 << (bit_width - 1))
    if kind == "minsi":
        return (1 << (bit_width - 1)) - 1
    return None


def _integer_bit_width(element_text: str) -> int | None:
    if not element_text.startswith("i") or not element_text[1:].isdecimal():
        return None
    return int(element_text[1:])


def _is_float_type(value_type: Type) -> bool:
    return str(value_type) in ("f16", "bf16", "f32", "f64")


def _scalar_dtype_text(source_dtype: str) -> str:
    head, separator, tail = source_dtype.rpartition("x")
    if separator and tail.isdecimal():
        return head
    return source_dtype


def _call_annotations(call: object) -> dict[str, object]:
    annotations = getattr(call, "annotations", None)
    return {str(key): value for key, value in mapping_items(annotations)}


def _static_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    integer = integer_value(value)
    if integer in (0, 1):
        return bool(integer)
    return None


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

_COPY_ANNOTATIONS = frozenset(
    (
        "barrier",
        "coalesced_width",
        "disable_tma",
        "emit_arrive",
        "eviction_policy",
        "force_cp_async",
        "is_async_copy",
        "is_tma_copy",
        "no_implicit_async_commit_wait",
        "parallel_loop_layout",
        "tl.pipeline_mbar_phase_expr",
        "use_2cta",
    )
)

_FILL_CALLS = {
    "tl.tileop.fill",
}

_REDUCE_CALLS = {
    "tl.tileop.reduce",
}

_FINALIZE_REDUCER_CALLS = {
    "tl.tileop.finalize_reducer",
}

_REDUCER_OPERATION_CODES = {
    "sum": 0,
    "max": 1,
    "min": 2,
}
