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

from loom.diagnostics import DiagnosticEngine
from loom.importers.core import (
    ImportBodyReport,
    ImportOptions,
    ImportResult,
    KernelArgumentSpec,
    KernelModuleShell,
    KernelModuleSpec,
    create_kernel_module,
    sanitize_symbol,
)
from loom.importers.tilelang.abi import extract_bindings
from loom.importers.tilelang.context import TileLangConversionContext
from loom.importers.tilelang.converter import TileLangConverter
from loom.importers.tilelang.defaults import build_default_registry
from loom.importers.tilelang.model import TileLangBinding, TileLangKernelFacts
from loom.importers.tilelang.nodes import (
    attrs,
    mapping_items,
    source_name,
)
from loom.importers.tilelang.ops.topology import collect_thread_extents, integer_value
from loom.importers.tilelang.types import TileLangTypeConverter
from loom.ir import Module, rebuild_value_metadata
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
    bindings = extract_bindings(prim_func, type_converter)
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
    for symbol, function in mapping_items(functions):
        name = source_name(symbol, fallback=str(symbol))
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
    source_attrs = attrs(prim_func)
    for key in ("global_symbol", "sym_name"):
        if key in source_attrs:
            return sanitize_symbol(str(source_attrs[key]), fallback="kernel")
    return sanitize_symbol(source_name(prim_func, fallback="main"), fallback="kernel")


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
        kernel_body_block=shell.body_block,
    )
    context.capture_existing_value_names()
    converter = TileLangConverter(build_default_registry())
    with shell.builder.insertion_block(shell.body_block):
        _map_kernel_arguments(shell, bindings, context)
        converter.convert_stmt(getattr(prim_func, "body", None), context)
        shell.builder.kernel.return_()
    rebuild_value_metadata(shell.module)
    return shell.module, context.finish(), converter.operation_counts


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
            for alias in binding.aliases:
                context.map_value(alias, argument, str(argument.type))
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
        data = getattr(binding.buffer, "data", None)
        if data is not None:
            context.map_value(data, view, str(view_type))
        context.record_converted(
            f"param {binding.name}",
            f"{context.ssa(view)} = buffer.view",
        )


def _workgroup_size(prim_func: object) -> tuple[int, int, int]:
    source_attrs = attrs(prim_func)
    body_extents = collect_thread_extents(getattr(prim_func, "body", None))
    thread_extent = source_attrs.get("thread_extent")
    thread_extent_items = mapping_items(thread_extent)
    if thread_extent_items or body_extents:
        extents = {str(key): value for key, value in body_extents.items()}
        for key, value in thread_extent_items:
            extent = integer_value(value)
            if extent is None:
                raise ValueError(f"TileLang thread extent `{key}` is not static")
            extents[str(key)] = extent
        return (
            int(extents.get("threadIdx.x", 1)),
            int(extents.get("threadIdx.y", 1)),
            int(extents.get("threadIdx.z", 1)),
        )
    launch = source_attrs.get("tir.kernel_launch_params")
    if isinstance(launch, Mapping):
        return (
            int(launch.get("block_dim_x", 1)),
            int(launch.get("block_dim_y", 1)),
            int(launch.get("block_dim_z", 1)),
        )
    return (1, 1, 1)
