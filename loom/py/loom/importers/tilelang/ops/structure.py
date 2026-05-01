# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Structural TileLang/TIR wrapper node converters."""

from __future__ import annotations

from collections.abc import Iterable

from loom.importers.tilelang.context import TileLangConversionContext
from loom.importers.tilelang.converter import (
    TileLangConverter,
    TileLangConverterRegistry,
)
from loom.importers.tilelang.nodes import node_text, source_name
from loom.importers.tilelang.ops.assumptions import (
    ASSUME_ATTR_KEYS,
    convert_assume_attr_stmt,
)
from loom.importers.tilelang.ops.memory import map_alloc_buffer
from loom.importers.tilelang.ops.topology import (
    map_thread_axis,
    mapped_thread_axis_sources,
    thread_axis_from_binding,
    thread_axis_name,
    thread_tag,
)


def register(registry: TileLangConverterRegistry) -> None:
    registry.register_statement("Block", convert_block)
    registry.register_statement("BlockRealize", convert_block_realize)
    registry.register_statement("LetStmt", convert_let_stmt)
    registry.register_statement("AttrStmt", convert_attr_stmt)


def convert_block(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Normalize a TIR block wrapper when it has no extra region semantics."""

    for buffer in tuple(getattr(stmt, "alloc_buffers", ()) or ()):
        if not map_alloc_buffer(buffer, context):
            return
    if _has_items(getattr(stmt, "match_buffers", ())):
        context.record_blocked(node_text(stmt), "block match buffers are not mapped")
        return
    init = getattr(stmt, "init", None)
    if init is not None:
        context.record_blocked(node_text(stmt), "block init region is not mapped")
        return
    converter.convert_stmt(getattr(stmt, "body", None), context)
    context.record_converted(node_text(stmt), "tir.Block normalized")


def convert_block_realize(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Normalize a TIR BlockRealize wrapper with a true predicate."""

    predicate = getattr(stmt, "predicate", None)
    if predicate is not None and not _is_true_predicate(predicate):
        context.record_blocked(node_text(stmt), "block predicate is not mapped")
        return
    converter.convert_stmt(getattr(stmt, "block", None), context)
    context.record_converted(node_text(stmt), "tir.BlockRealize normalized")


def convert_let_stmt(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Normalize a TIR let statement to an SSA value binding."""

    value = converter.convert_expr(getattr(stmt, "value", None), context)
    if value is None:
        context.record_blocked(node_text(stmt), "let value is not mapped")
        return
    var = getattr(stmt, "var", None)
    name = source_name(var, fallback="let")
    context.map_value(var, value, str(value.type))
    converter.convert_stmt(getattr(stmt, "body", None), context)
    context.record_converted(node_text(stmt), f"tir.LetStmt %{name} normalized")


def convert_attr_stmt(
    stmt: object,
    context: TileLangConversionContext,
    converter: TileLangConverter,
) -> None:
    """Normalize known metadata-only AttrStmt wrappers."""

    attr_key = str(getattr(stmt, "attr_key", ""))
    if attr_key in ASSUME_ATTR_KEYS:
        convert_assume_attr_stmt(stmt, context, converter)
        return
    if attr_key == "thread_extent":
        node = getattr(stmt, "node", None)
        thread_axis = thread_axis_from_binding(node)
        if thread_axis is None:
            if thread_tag(node) is not None:
                context.record_blocked(
                    node_text(stmt),
                    "thread_extent axis is not mapped",
                )
                return
            converter.convert_stmt(getattr(stmt, "body", None), context)
            context.record_converted(
                node_text(stmt),
                "AttrStmt `thread_extent` normalized",
            )
            return
        thread_value = map_thread_axis(
            axis=thread_axis,
            sources=mapped_thread_axis_sources(binding=node),
            context=context,
            converter=converter,
            name=thread_axis_name(node, thread_axis),
        )
        if thread_value is None:
            context.record_blocked(node_text(stmt), "thread_extent value is not mapped")
            return
        converter.convert_stmt(getattr(stmt, "body", None), context)
        operation_name = f"kernel.{thread_axis.loom_scope}.id<{thread_axis.dimension}>"
        context.record_converted(
            node_text(stmt),
            f"{context.ssa(thread_value)} = {operation_name}",
        )
        return
    if attr_key not in _NORMALIZED_ATTR_KEYS:
        context.record_blocked(node_text(stmt), f"AttrStmt `{attr_key}` is not mapped")
        return
    converter.convert_stmt(getattr(stmt, "body", None), context)
    context.record_converted(node_text(stmt), f"AttrStmt `{attr_key}` normalized")


def _has_items(value: object) -> bool:
    if value is None:
        return False
    if not isinstance(value, Iterable) or isinstance(value, str | bytes):
        return True
    return bool(tuple(value))


def _is_true_predicate(value: object) -> bool:
    if value is True:
        return True
    payload = getattr(value, "value", None)
    if payload is not None:
        return bool(payload)
    return bool(value)


_NORMALIZED_ATTR_KEYS = {
    "thread_extent",
}
