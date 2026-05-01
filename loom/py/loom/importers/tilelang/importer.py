# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Library-form TileLang kernel importer."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from loom.builder import ValueRef
from loom.diagnostics import DiagnosticEngine
from loom.importers.core import (
    ImportBodyReport,
    ImportOptions,
    ImportResult,
    KernelArgumentSpec,
    KernelModuleShell,
    KernelModuleSpec,
    SourceImportSession,
    create_kernel_module,
    sanitize_identifier,
    sanitize_symbol,
)
from loom.importers.tilelang.coverage import CoverageState, coverage_by_name
from loom.importers.tilelang.model import TileLangBinding, TileLangKernelFacts
from loom.importers.tilelang.types import TileLangTypeConverter
from loom.ir import BUFFER_TYPE, INDEX, OFFSET, Module, Type, rebuild_value_metadata
from loom.verify import verify_module


@dataclass(frozen=True, slots=True)
class TileLangImportOptions(ImportOptions):
    """Options specific to TileLang kernel imports."""

    kernel: str | None = None
    target_preset: str | None = None


@dataclass(frozen=True, slots=True)
class _NormalizedInput:
    source: object
    args: tuple[object, ...] = ()
    kwargs: Mapping[str, object] = field(default_factory=dict)
    target: str | None = None
    name: str | None = None


@dataclass(slots=True)
class TileLangConversionContext(SourceImportSession):
    """TileLang-specialized import session using Loom dynamic builders."""

    type_converter: TileLangTypeConverter = field(default_factory=TileLangTypeConverter)

    def type(self, value_type: str) -> Type:
        return self.type_converter.map_dtype(value_type)

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

    def fork(self, *, preview_block: object | None = None) -> TileLangConversionContext:
        child = TileLangConversionContext(
            builder=self.builder,
            diagnostics=self.diagnostics,
            preview_block=preview_block,
            value_map=dict(self.value_map),
            value_types=dict(self.value_types),
            constants=dict(self.constants),
            registry=self.registry,
            names=self.names,
            type_converter=self.type_converter,
        )
        return child


def import_tilelang(
    source: object,
    *,
    options: TileLangImportOptions | None = None,
) -> ImportResult:
    options = options or TileLangImportOptions()
    diagnostics = DiagnosticEngine()
    normalized = _normalize_input(source)
    resolved_source = _resolve_source(normalized)
    prim_func = _select_prim_func(
        resolved_source,
        options.kernel or normalized.name,
    )
    function_name = _function_name(prim_func, options.kernel or normalized.name)
    target_preset = options.target_preset or normalized.target or "tilelang.generic"
    type_converter = TileLangTypeConverter()
    bindings = _extract_bindings(prim_func, type_converter)
    workgroup_size = _workgroup_size(prim_func)
    loom_module, body_report, operation_counts = _build_loom_module(
        prim_func,
        bindings,
        function_name=function_name,
        target_preset=target_preset,
        workgroup_size=workgroup_size,
        diagnostics=diagnostics,
        type_converter=type_converter,
    )
    diagnostics.raise_if_errors()
    if options.verify_structure:
        verify_module(loom_module, diagnostics=diagnostics)
        diagnostics.raise_if_errors()
    facts = TileLangKernelFacts(
        function_name=function_name,
        target_preset=target_preset,
        workgroup_size=workgroup_size,
        bindings=bindings,
        operation_counts=operation_counts,
        converted_body=body_report,
    )
    return ImportResult(
        module=loom_module,
        diagnostics=diagnostics,
        report=facts if options.include_report else None,
    )


def import_tilelang_file(
    path: str | Path,
    *,
    options: TileLangImportOptions | None = None,
) -> ImportResult:
    raise ValueError(
        "TileLang imports require structured Python objects; "
        f"cannot import text file {path!s}"
    )


def _normalize_input(source: object) -> _NormalizedInput:
    if hasattr(source, "source") and type(source).__name__ == "TileLangImportInput":
        return _NormalizedInput(
            source=source.source,
            args=tuple(getattr(source, "args", ())),
            kwargs=dict(getattr(source, "kwargs", {})),
            target=getattr(source, "target", None),
            name=getattr(source, "name", None),
        )
    return _NormalizedInput(source=source)


def _resolve_source(normalized: _NormalizedInput) -> object:
    get_tir = getattr(normalized.source, "get_tir", None)
    if get_tir is None:
        return normalized.source
    return get_tir(*normalized.args, **dict(normalized.kwargs))


def _select_prim_func(source: object, requested_name: str | None) -> object:
    functions = getattr(source, "functions", None)
    if functions is None:
        return source
    matches: list[tuple[str, object]] = []
    for symbol, function in _mapping_items(functions):
        name = _source_name(symbol, fallback=str(symbol))
        if requested_name is None or name == requested_name:
            matches.append((name, function))
    if not matches:
        raise ValueError(f"TileLang module has no PrimFunc @{requested_name}")
    if len(matches) > 1:
        names = ", ".join(name for name, _function in matches)
        raise ValueError(f"TileLang module has multiple PrimFuncs: {names}")
    return matches[0][1]


def _function_name(prim_func: object, requested_name: str | None) -> str:
    if requested_name:
        return sanitize_symbol(requested_name, fallback="kernel")
    attrs = _attrs(prim_func)
    for key in ("global_symbol", "sym_name"):
        if key in attrs:
            return sanitize_symbol(str(attrs[key]), fallback="kernel")
    return sanitize_symbol(_source_name(prim_func, fallback="main"), fallback="kernel")


def _extract_bindings(
    prim_func: object,
    type_converter: TileLangTypeConverter,
) -> tuple[TileLangBinding, ...]:
    buffer_map = _buffer_map(prim_func)
    bindings: list[TileLangBinding] = []
    for ordinal, param in enumerate(tuple(getattr(prim_func, "params", ()))):
        buffer = _lookup_buffer(buffer_map, param)
        name = sanitize_identifier(_source_name(param, fallback=f"arg{ordinal}"))
        if buffer is None:
            value_type = type_converter.map_dtype(_dtype(param), index_like=False)
        else:
            value_type = BUFFER_TYPE
        bindings.append(
            TileLangBinding(
                ordinal=ordinal,
                name=name,
                source=param,
                type=value_type,
                buffer=buffer,
            )
        )
    if not bindings:
        raise ValueError("TileLang PrimFunc has no parameters")
    return tuple(bindings)


def _build_loom_module(
    prim_func: object,
    bindings: tuple[TileLangBinding, ...],
    *,
    function_name: str,
    target_preset: str,
    workgroup_size: tuple[int, int, int],
    diagnostics: DiagnosticEngine,
    type_converter: TileLangTypeConverter,
) -> tuple[Module, ImportBodyReport, Counter[str]]:
    shell = create_kernel_module(
        KernelModuleSpec(
            target_preset=target_preset,
            target_symbol=sanitize_symbol(target_preset, fallback="target"),
            export_symbol=function_name,
            callee=function_name,
            workgroup_size=workgroup_size,
            arguments=[
                KernelArgumentSpec(
                    ordinal=binding.ordinal,
                    name=binding.name,
                    type=binding.type,
                )
                for binding in bindings
            ],
        )
    )
    context = TileLangConversionContext(
        builder=shell.builder,
        diagnostics=diagnostics,
        preview_block=shell.body_block,
        type_converter=type_converter,
    )
    context.capture_existing_value_names()
    operation_counts: Counter[str] = Counter()
    with shell.builder.insertion_block(shell.body_block):
        _map_kernel_arguments(shell, bindings, context)
        _convert_stmt(getattr(prim_func, "body", None), context, operation_counts)
        shell.builder.kernel.return_()
    rebuild_value_metadata(shell.module)
    return shell.module, context.finish(), operation_counts


def _map_kernel_arguments(
    shell: KernelModuleShell,
    bindings: tuple[TileLangBinding, ...],
    context: TileLangConversionContext,
) -> None:
    zero = context.ensure_constant("0", "offset", "c0_bytes")
    for binding in bindings:
        argument = shell.arguments_by_ordinal[binding.ordinal]
        if binding.buffer is None:
            context.map_value(binding.source, argument, str(argument.type))
            continue
        view_type = context.type_converter.view_type(binding.buffer)
        view = context.builder.buffer.view(
            buffer=argument,
            byte_offset=zero,
            results=[view_type],
            name=binding.name,
        )
        context.map_value(binding.source, view, str(view_type))
        context.map_value(binding.buffer, view, str(view_type))
        context.record_converted(
            f"param {binding.name}",
            f"{context.ssa(view)} = buffer.view",
        )


def _convert_stmt(
    stmt: object,
    context: TileLangConversionContext,
    operation_counts: Counter[str],
) -> None:
    if stmt is None:
        return
    kind = _node_kind(stmt)
    operation_counts[kind] += 1
    if kind == "SeqStmt":
        for child in tuple(getattr(stmt, "seq", ())):
            _convert_stmt(child, context, operation_counts)
        return
    if kind == "BufferStore":
        _convert_buffer_store(stmt, context, operation_counts)
        return
    if kind == "For":
        _convert_for(stmt, context, operation_counts)
        return
    if kind == "IfThenElse":
        _convert_if(stmt, context, operation_counts)
        return
    if kind == "Evaluate":
        value = getattr(stmt, "value", None)
        if value is not None:
            _convert_expr(value, context, operation_counts)
        return
    _record_unsupported(stmt, context)


def _convert_buffer_store(
    stmt: object,
    context: TileLangConversionContext,
    operation_counts: Counter[str],
) -> None:
    buffer = getattr(stmt, "buffer", None)
    view = context.mapped(buffer)
    if view is None:
        context.record_blocked(_node_text(stmt), "buffer store target is not mapped")
        return
    value = _convert_expr(getattr(stmt, "value", None), context, operation_counts)
    indices = [
        _convert_expr(index, context, operation_counts, index_like=True)
        for index in tuple(getattr(stmt, "indices", ()))
    ]
    if value is None or any(index is None for index in indices):
        context.record_blocked(_node_text(stmt), "buffer store operands are not mapped")
        return
    context.builder.view.store(
        value=value,
        view=view,
        indices=[index for index in indices if index is not None],
    )
    context.record_converted(_node_text(stmt), "view.store")


def _convert_for(
    stmt: object,
    context: TileLangConversionContext,
    operation_counts: Counter[str],
) -> None:
    loop_var = getattr(stmt, "loop_var", None)
    loop_name = sanitize_identifier(_source_name(loop_var, fallback="iv"))
    lower = _convert_expr(
        getattr(stmt, "min", None), context, operation_counts, index_like=True
    )
    extent = _convert_expr(
        getattr(stmt, "extent", None),
        context,
        operation_counts,
        index_like=True,
    )
    if lower is None or extent is None:
        context.record_blocked(_node_text(stmt), "loop bounds are not mapped")
        return
    upper = context.builder.index.add(
        lhs=lower,
        rhs=extent,
        results=[INDEX],
        name=context.fresh_name(f"{loop_name}_ub"),
    )
    step = context.ensure_constant("1", "index", "c1")
    body = context.builder.region(args=[(loop_name, INDEX)])
    loop_var_ref = ValueRef(body.blocks[0].arg_ids[0], context.builder.ir)
    child = context.fork(preview_block=body.blocks[0])
    child.map_value(loop_var, loop_var_ref, "index")
    with context.builder.insertion_block(body.blocks[0]):
        _convert_stmt(getattr(stmt, "body", None), child, operation_counts)
        context.builder.scf.yield_()
    context.merge_child_records(child)
    context.builder.scf.for_(
        lower_bound=lower,
        upper_bound=upper,
        step=step,
        iter_args=[],
        results=[],
        body=body,
    )
    context.record_converted(_node_text(stmt), "scf.for")


def _convert_if(
    stmt: object,
    context: TileLangConversionContext,
    operation_counts: Counter[str],
) -> None:
    condition = _convert_expr(
        getattr(stmt, "condition", None), context, operation_counts
    )
    if condition is None:
        context.record_blocked(_node_text(stmt), "if condition is not mapped")
        return
    then_region = context.builder.region()
    else_region = context.builder.region()
    then_child = context.fork(preview_block=then_region.blocks[0])
    else_child = context.fork(preview_block=else_region.blocks[0])
    with context.builder.insertion_block(then_region.blocks[0]):
        _convert_stmt(getattr(stmt, "then_case", None), then_child, operation_counts)
        context.builder.scf.yield_()
    with context.builder.insertion_block(else_region.blocks[0]):
        _convert_stmt(getattr(stmt, "else_case", None), else_child, operation_counts)
        context.builder.scf.yield_()
    context.merge_child_records(then_child)
    context.merge_child_records(else_child)
    context.builder.scf.if_(
        condition=condition,
        results=[],
        then_region=then_region,
        else_region=else_region,
    )
    context.record_converted(_node_text(stmt), "scf.if")


def _convert_expr(
    expr: object,
    context: TileLangConversionContext,
    operation_counts: Counter[str],
    *,
    index_like: bool = False,
) -> ValueRef | None:
    if expr is None:
        return None
    mapped = context.mapped(expr)
    if mapped is not None:
        return mapped
    kind = _node_kind(expr)
    operation_counts[kind] += 1
    if kind in {"IntImm", "FloatImm"}:
        value = getattr(expr, "value", expr)
        value_type = "index" if index_like else _dtype(expr)
        name = context.fresh_name("c" if index_like else "const")
        result = context.build_constant(value, value_type, name)
        context.map_value(expr, result, value_type)
        return result
    if kind == "Var":
        mapped_var = context.mapped(expr)
        if mapped_var is None:
            context.record_blocked(_node_text(expr), "variable is not mapped")
        return mapped_var
    if kind == "BufferLoad":
        return _convert_buffer_load(expr, context, operation_counts)
    if kind in _BINARY_INDEX_OPS:
        return _convert_binary_expr(
            expr,
            context,
            operation_counts,
            index_like=index_like,
        )
    _record_unsupported(expr, context)
    return None


def _convert_buffer_load(
    expr: object,
    context: TileLangConversionContext,
    operation_counts: Counter[str],
) -> ValueRef | None:
    buffer = getattr(expr, "buffer", None)
    view = context.mapped(buffer)
    if view is None:
        context.record_blocked(_node_text(expr), "buffer load source is not mapped")
        return None
    indices = [
        _convert_expr(index, context, operation_counts, index_like=True)
        for index in tuple(getattr(expr, "indices", ()))
    ]
    if any(index is None for index in indices):
        context.record_blocked(_node_text(expr), "buffer load indices are not mapped")
        return None
    result_type = context.type_converter.map_dtype(_dtype(expr))
    result = context.builder.view.load(
        view=view,
        indices=[index for index in indices if index is not None],
        results=[result_type],
        name=context.fresh_name("load"),
    )
    context.map_value(expr, result, str(result_type))
    context.record_converted(_node_text(expr), f"{context.ssa(result)} = view.load")
    return result


def _convert_binary_expr(
    expr: object,
    context: TileLangConversionContext,
    operation_counts: Counter[str],
    *,
    index_like: bool,
) -> ValueRef | None:
    lhs = _convert_expr(
        getattr(expr, "a", None), context, operation_counts, index_like=index_like
    )
    rhs = _convert_expr(
        getattr(expr, "b", None), context, operation_counts, index_like=index_like
    )
    if lhs is None or rhs is None:
        context.record_blocked(_node_text(expr), "binary operands are not mapped")
        return None
    kind = _node_kind(expr)
    if index_like:
        builder_name = _BINARY_INDEX_OPS[kind]
        builder = getattr(context.builder.index, builder_name)
        result = cast(
            ValueRef,
            builder(
                lhs=lhs,
                rhs=rhs,
                results=[INDEX],
                name=context.fresh_name(builder_name),
            ),
        )
        context.map_value(expr, result, "index")
        return result
    result_type = context.type_converter.map_dtype(_dtype(expr))
    if str(result_type).startswith("f"):
        scalar_builder_name = _BINARY_FLOAT_OPS.get(kind)
    else:
        scalar_builder_name = _BINARY_INTEGER_OPS.get(kind)
    if scalar_builder_name is None:
        context.record_blocked(_node_text(expr), f"no scalar builder for {kind}")
        return None
    builder = getattr(context.builder.scalar, scalar_builder_name)
    result = cast(
        ValueRef,
        builder(
            lhs=lhs,
            rhs=rhs,
            results=[result_type],
            name=context.fresh_name(scalar_builder_name),
        ),
    )
    context.map_value(expr, result, str(result_type))
    return result


def _record_unsupported(node: object, context: TileLangConversionContext) -> None:
    kind = _node_kind(node)
    row = coverage_by_name().get(f"tir.{kind}") or coverage_by_name().get(kind)
    if row is None:
        context.record_unsupported(kind, _node_text(node))
        return
    if row.state == CoverageState.SUPPORTED:
        reason = "coverage manifest marks this node supported but no converter ran"
    else:
        reason = f"coverage state is {row.state.value}: {row.note}"
    context.record_blocked(_node_text(node), reason)


def _workgroup_size(prim_func: object) -> tuple[int, int, int]:
    attrs = _attrs(prim_func)
    thread_extent = attrs.get("thread_extent")
    if isinstance(thread_extent, Mapping):
        return (
            int(thread_extent.get("threadIdx.x", 1)),
            int(thread_extent.get("threadIdx.y", 1)),
            int(thread_extent.get("threadIdx.z", 1)),
        )
    launch = attrs.get("tir.kernel_launch_params")
    if isinstance(launch, Mapping):
        return (
            int(launch.get("block_dim_x", 1)),
            int(launch.get("block_dim_y", 1)),
            int(launch.get("block_dim_z", 1)),
        )
    return (1, 1, 1)


def _buffer_map(prim_func: object) -> dict[object, object]:
    return dict(_mapping_items(getattr(prim_func, "buffer_map", {})))


def _lookup_buffer(buffer_map: Mapping[object, object], param: object) -> object | None:
    if param in buffer_map:
        return buffer_map[param]
    param_name = _source_name(param, fallback=str(param))
    for key, value in buffer_map.items():
        if _source_name(key, fallback=str(key)) == param_name:
            return value
    return None


def _attrs(value: object) -> Mapping[str, object]:
    attrs = getattr(value, "attrs", {})
    if attrs is None:
        return {}
    return {str(key): item for key, item in _mapping_items(attrs)}


def _mapping_items(value: object) -> tuple[tuple[object, object], ...]:
    items = getattr(value, "items", None)
    if items is None:
        return ()
    return tuple(items())


def _dtype(value: object) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return "int32"
    return str(dtype)


def _source_name(value: object, *, fallback: str) -> str:
    for attr in ("name_hint", "name", "__name__"):
        name = getattr(value, attr, None)
        if name:
            return str(name)
    return fallback


def _node_kind(node: object) -> str:
    return type(node).__name__


def _node_text(node: object) -> str:
    return str(node).splitlines()[0]


_BINARY_INDEX_OPS = {
    "Add": "add",
    "Sub": "sub",
    "Mul": "mul",
    "FloorDiv": "div",
    "FloorMod": "rem",
}

_BINARY_INTEGER_OPS = {
    "Add": "addi",
    "Sub": "subi",
    "Mul": "muli",
    "FloorDiv": "floordivsi",
    "FloorMod": "remsi",
    "Div": "divsi",
}

_BINARY_FLOAT_OPS = {
    "Add": "addf",
    "Sub": "subf",
    "Mul": "mulf",
    "Div": "divf",
}
