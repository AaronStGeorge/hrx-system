# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TileLang tileop call converters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from loom.builder import ValueRef
from loom.importers.tilelang.buffers import (
    TileLangBufferAccess,
    TileLangBufferRegion,
    map_region_indices,
    resolve_buffer_access,
    resolve_buffer_region,
)
from loom.importers.tilelang.context import TileLangConversionContext
from loom.importers.tilelang.converter import ExpressionOptions, TileLangConverter
from loom.importers.tilelang.coverage import coverage_row
from loom.importers.tilelang.nodes import dtype, mapping_items, node_kind, node_text
from loom.importers.tilelang.ops.topology import integer_value
from loom.ir import INDEX, ShapedType, StaticDim, Type, TypeKind


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
    if op_name in _GEMM_CALLS:
        return _convert_gemm_call(expr, context, options, op_name)
    _record_unsupported_tileop(expr, context, op_name)
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
    loop_extents = _copy_loop_extents(source, target, context)
    if loop_extents is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` source and destination extents differ",
        )
        return None
    source_element_type = _view_element_type(source.view, context)
    if source_element_type is None:
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

    def emit_copy(
        indices: tuple[ValueRef, ...],
        body_context: TileLangConversionContext,
    ) -> None:
        source_indices = _project_region_indices(source, indices, expr, body_context)
        target_indices = _project_region_indices(target, indices, expr, body_context)
        if source_indices is None or target_indices is None:
            return
        value = body_context.builder.view.load(
            view=source.view,
            indices=list(source_indices),
            results=[source_element_type],
            name=body_context.fresh_name("copy"),
        )
        converted_value = _copy_value_cast(
            value,
            source_element_type,
            target_element_type,
            expr,
            body_context,
        )
        if converted_value is None:
            return
        body_context.builder.view.store(
            value=converted_value,
            view=target.view,
            indices=list(target_indices),
        )

    if not _emit_region_loops(loop_extents, context, emit_copy):
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
            TileLangBufferAccess(target.view, target_indices),
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
        return _convert_finalize_reducer_all(
            expr,
            context,
            converter,
            op_name,
            args=args,
            operation=reducer_info.operation,
            data_source=data_source,
        )
    context.record_blocked(
        node_text(expr),
        (
            "tl.tileop.finalize_reducer replication "
            f"`{reducer_info.replication}` is not imported"
        ),
    )
    return None


def _convert_finalize_reducer_all(
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    op_name: str,
    *,
    args: tuple[object, ...],
    operation: str,
    data_source: object,
) -> ValueRef | None:
    target = _decode_region_arg(
        args=args,
        position=0,
        expr=expr,
        context=context,
        converter=converter,
    )
    if target is None:
        return None
    element_type = _view_element_type(target.view, context)
    if element_type is None:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` reducer region is not a shaped view",
        )
        return None
    source_dtype = _reducer_source_dtype(data_source, context)
    reduce_spec = _reducer_reduce_spec(operation, source_dtype, element_type)
    if reduce_spec is None:
        context.record_blocked(
            node_text(expr),
            (
                f"tl.tileop.finalize_reducer operation `{operation}` "
                f"is not imported for {element_type}"
            ),
        )
        return None

    def emit_reduce(
        indices: tuple[ValueRef, ...],
        body_context: TileLangConversionContext,
    ) -> None:
        target_indices = _region_indices(target, indices, body_context)
        value = body_context.builder.view.load(
            view=target.view,
            indices=list(target_indices),
            results=[element_type],
            name=body_context.fresh_name("reducer_value"),
        )
        result = body_context.builder.kernel.workgroup_reduce(
            kind=reduce_spec.combiner,
            value=value,
            results=[element_type],
            name=body_context.fresh_name("reducer_all"),
        )
        body_context.builder.view.store(
            value=result,
            view=target.view,
            indices=list(target_indices),
        )

    if not _emit_region_loops(target.extents, context, emit_reduce):
        context.record_blocked(
            node_text(expr), f"call `{op_name}` reducer region is not mapped"
        )
        return None
    context.record_converted(
        node_text(expr),
        f"tl.tileop.finalize_reducer<{operation}, all>",
    )
    return None


def _convert_gemm_call(
    expr: object,
    context: TileLangConversionContext,
    options: ExpressionOptions,
    op_name: str,
) -> ValueRef | None:
    if not _require_effect(expr, context, options, op_name):
        return None
    args = _args(expr)
    if len(args) < 19:
        context.record_blocked(
            node_text(expr),
            f"call `{op_name}` expects at least 19 descriptor operands",
        )
        return None
    descriptor = ", ".join(
        (
            f"a={_region_source_name(args[0])}",
            f"b={_region_source_name(args[1])}",
            f"c={_region_source_name(args[2])}",
            f"transpose_a={_field_bool(args[3])}",
            f"transpose_b={_field_bool(args[4])}",
            f"m={_field_integer(args[5])}",
            f"n={_field_integer(args[6])}",
            f"k={_field_integer(args[7])}",
            f"policy={_field_integer(args[8])}",
            f"clear_accum={_field_bool(args[9])}",
            f"k_pack={_field_integer(args[14])}",
            f"wg_wait={_field_integer(args[15])}",
        )
    )
    row = coverage_row(op_name)
    if row is None:
        context.record_blocked(
            node_text(expr),
            f"TileLang GEMM operation `{op_name}` is not in the coverage manifest",
        )
        return None
    reason = f"call `{op_name}` coverage state is {row.state.value}: {row.note}"
    context.record_blocked(node_text(expr), f"{reason} ({descriptor})")
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


def _record_unsupported_tileop(
    expr: object,
    context: TileLangConversionContext,
    op_name: str,
) -> None:
    row = coverage_row(op_name)
    if row is None:
        context.record_blocked(
            node_text(expr),
            f"TileLang tile operation `{op_name}` is not in the coverage manifest",
        )
        return
    context.record_blocked(
        node_text(expr),
        f"call `{op_name}` coverage state is {row.state.value}: {row.note}",
    )


def _decode_region(
    expr: object,
    owner: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
    *,
    expected_access: int,
) -> TileLangBufferRegion | None:
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
    return resolve_buffer_region(
        getattr(buffer_load, "buffer", None),
        tuple(getattr(buffer_load, "indices", ())),
        args[2:],
        context,
        converter,
        diagnostic_owner=owner,
    )


def _decode_region_arg(
    *,
    args: tuple[object, ...],
    position: int,
    expr: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> TileLangBufferRegion | None:
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
) -> TileLangBufferRegion | None:
    ranges = tuple(getattr(expr, "region", ()))
    return resolve_buffer_region(
        getattr(expr, "buffer", None),
        tuple(getattr(item, "min", None) for item in ranges),
        tuple(getattr(item, "extent", None) for item in ranges),
        context,
        converter,
        diagnostic_owner=owner,
    )


def _decode_buffer_load_region(
    expr: object,
    owner: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> TileLangBufferRegion | None:
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
    return resolve_buffer_region(
        getattr(expr, "buffer", None),
        tuple(indices),
        tuple(extents),
        context,
        converter,
        diagnostic_owner=owner,
    )


def _decode_buffer_access(
    expr: object,
    owner: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> TileLangBufferAccess | None:
    if node_kind(expr) != "BufferLoad":
        context.record_blocked(
            node_text(owner),
            f"tile operation buffer access must be BufferLoad, got `{node_kind(expr)}`",
        )
        return None
    return resolve_buffer_access(
        getattr(expr, "buffer", None),
        tuple(getattr(expr, "indices", ())),
        context,
        converter,
        diagnostic_owner=owner,
    )


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


def _copy_loop_extents(
    source: TileLangBufferRegion,
    target: TileLangBufferRegion,
    context: TileLangConversionContext,
) -> tuple[ValueRef, ...] | None:
    source_extents = _squeezed_singleton_extents(source.extents, context)
    target_extents = _squeezed_singleton_extents(target.extents, context)
    if len(source_extents) != len(target_extents):
        return None
    if all(
        lhs.id == rhs.id
        for lhs, rhs in zip(source_extents, target_extents, strict=False)
    ):
        return source_extents
    return None


def _squeezed_singleton_extents(
    extents: tuple[ValueRef, ...],
    context: TileLangConversionContext,
) -> tuple[ValueRef, ...]:
    return tuple(
        extent for extent in extents if not _is_static_one_extent(extent, context)
    )


def _is_static_one_extent(
    extent: ValueRef,
    context: TileLangConversionContext,
) -> bool:
    one = context.constants.get(("index", "1"))
    return one is not None and extent.id == one.id


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
    region: TileLangBufferRegion,
    loop_indices: tuple[ValueRef, ...],
    context: TileLangConversionContext,
) -> tuple[ValueRef, ...]:
    return map_region_indices(region, loop_indices, context)


def _project_region_indices(
    region: TileLangBufferRegion,
    loop_indices: tuple[ValueRef, ...],
    owner: object,
    context: TileLangConversionContext,
) -> tuple[ValueRef, ...] | None:
    zero = context.ensure_constant("0", "index", "c0")
    offsets: list[ValueRef] = []
    loop_index = 0
    for extent in region.extents:
        if _is_static_one_extent(extent, context):
            offsets.append(zero)
            continue
        if loop_index >= len(loop_indices):
            context.record_blocked(
                node_text(owner),
                "tile operation copy projected rank does not match loop rank",
            )
            return None
        offsets.append(loop_indices[loop_index])
        loop_index += 1
    if loop_index != len(loop_indices):
        context.record_blocked(
            node_text(owner),
            "tile operation copy loop rank does not match projected rank",
        )
        return None
    return _region_indices(region, tuple(offsets), context)


def _reduction_target_indices(
    target: TileLangBufferRegion,
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
        zero = context.ensure_constant("0", "index", "c0")
        return _region_indices(target, (zero,), context)
    return None


def _emit_scalar_reduce_loop(
    source: TileLangBufferRegion,
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
    source: TileLangBufferRegion,
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


def _copy_value_cast(
    value: ValueRef,
    source_element_type: Type,
    target_element_type: Type,
    owner: object,
    context: TileLangConversionContext,
) -> ValueRef | None:
    if str(source_element_type) == str(target_element_type):
        return value
    source_width = _integer_bit_width(str(source_element_type))
    target_width = _integer_bit_width(str(target_element_type))
    if source_width is not None and target_width is not None:
        if source_width < target_width:
            return context.builder.scalar.extsi(
                input=value,
                results=[target_element_type],
                name=context.fresh_name("copy_ext"),
            )
        if source_width > target_width:
            return context.builder.scalar.trunci(
                input=value,
                results=[target_element_type],
                name=context.fresh_name("copy_trunc"),
            )
    context.record_blocked(
        node_text(owner),
        (
            "tl.tileop.copy element conversion "
            f"{source_element_type} to {target_element_type} is not imported"
        ),
    )
    return None


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


def _reducer_reduce_spec(
    operation: str,
    source_dtype: str | None,
    element_type: Type,
) -> TileReduceSpec | None:
    element_text = str(element_type)
    is_float = _is_float_type(element_type)
    is_unsigned = source_dtype is not None and _scalar_dtype_text(
        source_dtype
    ).startswith(("uint", "u"))
    if operation == "sum":
        return TileReduceSpec(operation, "addf" if is_float else "addi")
    if operation == "max":
        if is_float:
            return TileReduceSpec(operation, "maxnumf")
        if _integer_bit_width(element_text) is not None:
            return TileReduceSpec(operation, "maxui" if is_unsigned else "maxsi")
    if operation == "min":
        if is_float:
            return TileReduceSpec(operation, "minnumf")
        if _integer_bit_width(element_text) is not None:
            return TileReduceSpec(operation, "minui" if is_unsigned else "minsi")
    return None


def _reducer_source_dtype(
    data_source: object,
    context: TileLangConversionContext,
) -> str | None:
    buffer = context.mapped_buffer_for_data(data_source)
    return None if buffer is None else dtype(buffer)


def _reduce_init(
    clear: bool,
    reduce_spec: TileReduceSpec,
    element_type: Type,
    target: TileLangBufferAccess,
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
    is_float = _is_float_type(element_type)
    if reduce_spec.source_kind == "absmax":
        return 0.0 if is_float else 0
    kind = reduce_spec.combiner
    element_text = str(element_type)
    bit_width = _integer_bit_width(element_text)
    if kind == "addf":
        return 0.0
    if kind in ("addi", "ori", "xori"):
        return 0
    if kind == "mulf":
        return 1.0
    if kind == "muli":
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


def _field_bool(value: object) -> str:
    boolean = _static_bool(value)
    if boolean is None:
        return "dynamic"
    return "true" if boolean else "false"


def _field_integer(value: object) -> str:
    integer = integer_value(value)
    if integer is None:
        return "dynamic"
    return str(integer)


def _region_source_name(value: object) -> str:
    if _call_op_name(value) == "tl.tileop.region":
        args = _args(value)
        if args:
            return _region_source_name(args[0])
    if node_kind(value) == "BufferLoad":
        buffer = getattr(value, "buffer", None)
        name = getattr(buffer, "name", None)
        if name:
            return str(name)
        data = getattr(buffer, "data", None)
        data_name = getattr(data, "name", None)
        if data_name:
            return str(data_name)
    return node_kind(value)


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

_GEMM_CALLS = {
    "tl.tileop.gemm",
}

_REDUCER_OPERATION_CODES = {
    "sum": 0,
    "max": 1,
    "min": 2,
}
